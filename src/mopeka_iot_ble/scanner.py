"""Async scanner for Mopeka Standard BLE sensors.

Provides high-level async API for discovering and reading Standard sensors.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

from .models import MopekaStandardDevice, SensorReading
from .parser import (
    MOPEKA_STANDARD_MANUFACTURER,
    MOPEKA_STANDARD_SERVICE_UUID,
    calculate_lpg_speed_of_sound,
    calculate_distance_mm,
    calculate_tank_level_percentage,
    convert_standard_temperature,
    convert_standard_battery,
    find_strongest_reflection,
)

_LOGGER = logging.getLogger(__name__)


class MopekaStandardScanner:
    """Async scanner for Mopeka Standard BLE sensors.

    Provides methods for device discovery, sensor reading, and callback registration.
    Supports async context manager pattern for proper resource management.
    """

    def __init__(self) -> None:
        """Initialize the scanner."""
        self._callbacks: Dict[str, List[Callable[[SensorReading], None]]] = {}
        self._scanner: Optional[BleakScanner] = None
        self._scanning: bool = False
        self._scan_lock = asyncio.Lock()

    async def __aenter__(self) -> MopekaStandardScanner:
        """Async context manager entry."""
        self._scanner = BleakScanner()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._scanning and self._scanner:
            await self._scanner.stop()
        self._scanner = None
        self._scanning = False

    def register_callback(
        self, mac_address: str, callback: Callable[[SensorReading], None]
    ) -> None:
        """Register a callback for sensor readings from a specific device.

        Args:
            mac_address: MAC address of the device to monitor
            callback: Function to call with SensorReading when data is received
        """
        if mac_address not in self._callbacks:
            self._callbacks[mac_address] = []
        self._callbacks[mac_address].append(callback)

    def unregister_callback(
        self, mac_address: str, callback: Callable[[SensorReading], None]
    ) -> None:
        """Unregister a callback for a specific device.

        Args:
            mac_address: MAC address of the device
            callback: The callback function to remove
        """
        if mac_address in self._callbacks:
            try:
                self._callbacks[mac_address].remove(callback)
                if not self._callbacks[mac_address]:
                    del self._callbacks[mac_address]
            except ValueError:
                pass  # Callback wasn't registered

    def _is_standard_sensor(self, device: BLEDevice, adv_data: AdvertisementData) -> bool:
        """Check if device is a Mopeka Standard sensor.

        Args:
            device: The BLE device
            adv_data: Advertisement data

        Returns:
            True if device is a Standard sensor
        """
        # Check for Standard service UUID
        if MOPEKA_STANDARD_SERVICE_UUID not in adv_data.service_uuids:
            return False

        # Check for Standard manufacturer data
        if MOPEKA_STANDARD_MANUFACTURER not in adv_data.manufacturer_data:
            return False

        # Verify manufacturer data length (23 bytes)
        manufacturer_data = adv_data.manufacturer_data[MOPEKA_STANDARD_MANUFACTURER]
        if len(manufacturer_data) != 23:
            return False

        return True

    async def scan_for_devices(self, timeout: float = 10.0) -> List[Dict[str, Any]]:
        """Scan for Mopeka Standard sensors and return discovered devices.

        Args:
            timeout: Scan timeout in seconds (default: 10.0)

        Returns:
            List of discovered device information dictionaries containing:
            - mac_address: Device MAC address
            - device_name: Device name from advertisement
            - signal_strength: RSSI signal strength in dBm
            - manufacturer_data: Raw manufacturer data bytes

        Raises:
            RuntimeError: If scanner is not initialized or scan fails
        """
        if not self._scanner:
            raise RuntimeError("Scanner not initialized. Use async context manager.")

        discovered_devices = []

        async with self._scan_lock:
            try:
                _LOGGER.debug(f"Starting BLE scan for Standard sensors (timeout: {timeout}s)")
                devices = await BleakScanner.discover(timeout=timeout, return_adv=True)

                for device, adv_data in devices.values():
                    if self._is_standard_sensor(device, adv_data):
                        device_info = {
                            "mac_address": device.address,
                            "device_name": device.name or f"Mopeka Standard {device.address[-5:]}",
                            "signal_strength": adv_data.rssi,
                            "manufacturer_data": adv_data.manufacturer_data[MOPEKA_STANDARD_MANUFACTURER],
                        }
                        discovered_devices.append(device_info)
                        _LOGGER.debug(f"Found Standard sensor: {device.address} (RSSI: {adv_data.rssi})")

                _LOGGER.info(f"Discovered {len(discovered_devices)} Standard sensor(s)")
                return discovered_devices

            except Exception as e:
                _LOGGER.error(f"BLE scan failed: {e}")
                raise RuntimeError(f"Failed to scan for devices: {e}") from e

    async def read_sensor(
        self, 
        mac_address: str, 
        device_config: MopekaStandardDevice,
        timeout: float = 5.0
    ) -> Optional[SensorReading]:
        """Read sensor data from a specific Mopeka Standard sensor.

        Args:
            mac_address: MAC address of the device to read
            device_config: Device configuration with tank geometry and mixture settings
            timeout: Scan timeout in seconds (default: 5.0)

        Returns:
            SensorReading object with parsed data, or None if device not found

        Raises:
            RuntimeError: If scanner is not initialized or scan fails
            ValueError: If device configuration is invalid
        """
        if not self._scanner:
            raise RuntimeError("Scanner not initialized. Use async context manager.")

        # Validate device configuration
        if mac_address.upper() != device_config.mac_address.upper():
            raise ValueError("MAC address mismatch between parameter and device config")

        async with self._scan_lock:
            try:
                _LOGGER.debug(f"Scanning for specific device: {mac_address}")
                devices = await BleakScanner.discover(timeout=timeout, return_adv=True)

                for device, adv_data in devices.values():
                    if (device.address.upper() == mac_address.upper() and 
                        self._is_standard_sensor(device, adv_data)):
                        
                        # Parse the manufacturer data
                        manufacturer_data = adv_data.manufacturer_data[MOPEKA_STANDARD_MANUFACTURER]
                        parsed_data = self._parse_standard_data(manufacturer_data)
                        
                        # Convert raw values to physical measurements
                        temperature_celsius = convert_standard_temperature(parsed_data['raw_temp'])
                        battery_voltage, battery_percentage = convert_standard_battery(parsed_data['raw_voltage'])
                        
                        # Calculate tank level if measurements available
                        tank_level_mm = None
                        tank_level_percentage = None
                        
                        if parsed_data['measurements_time'] and parsed_data['measurements_value']:
                            # Create measurement tuples for strongest reflection analysis
                            measurements = list(zip(
                                parsed_data['measurements_time'], 
                                parsed_data['measurements_value']
                            ))
                            
                            # Find strongest reflection
                            strongest = find_strongest_reflection(measurements)
                            if strongest:
                                time_value, amplitude = strongest
                                
                                # Calculate distance using LPG speed of sound
                                speed_of_sound = calculate_lpg_speed_of_sound(
                                    temperature_celsius, 
                                    device_config.propane_butane_mix
                                )
                                tank_level_mm = calculate_distance_mm(time_value, speed_of_sound)
                                
                                # Calculate tank level percentage
                                tank_level_percentage = calculate_tank_level_percentage(
                                    tank_level_mm,
                                    device_config.empty_distance_mm,
                                    device_config.full_distance_mm
                                )

                        # Create sensor reading
                        reading = SensorReading(
                            mac_address=device.address,
                            tank_level_mm=tank_level_mm,
                            tank_level_percentage=tank_level_percentage,
                            temperature_celsius=temperature_celsius,
                            battery_percentage=battery_percentage,
                            battery_voltage=battery_voltage,
                            sync_pressed=parsed_data['sync_pressed'],
                            slow_update_rate=parsed_data['slow_update_rate'],
                            timestamp=datetime.now(),
                            signal_strength=adv_data.rssi
                        )

                        # Trigger callbacks for this device
                        if mac_address in self._callbacks:
                            for callback in self._callbacks[mac_address]:
                                try:
                                    callback(reading)
                                except Exception as e:
                                    _LOGGER.error(f"Callback error for {mac_address}: {e}")

                        _LOGGER.debug(f"Successfully read sensor data from {mac_address}")
                        return reading

                _LOGGER.warning(f"Device {mac_address} not found during scan")
                return None

            except Exception as e:
                _LOGGER.error(f"Failed to read sensor {mac_address}: {e}")
                raise RuntimeError(f"Failed to read sensor: {e}") from e

    def _parse_standard_data(self, data: bytes) -> Dict[str, Any]:
        """Parse 23-byte Standard sensor manufacturer data structure.
        
        This is a standalone version of the parser logic for use in the scanner.
        
        Args:
            data: 23-byte manufacturer data
            
        Returns:
            Dictionary with parsed sensor data
        """
        import struct
        
        # Parse first 4 bytes: device_id (2 bytes), raw_voltage (1 byte), temp_and_flags (1 byte)
        device_id, raw_voltage, temp_and_flags = struct.unpack('<HBB', data[:4])
        
        # Extract flags and temperature from temp_and_flags byte
        sync_pressed = bool(temp_and_flags & 0x80)
        slow_update_rate = bool(temp_and_flags & 0x40)
        raw_temp = temp_and_flags & 0x3F
        
        # Extract measurement data from bytes 4-22 (19 bytes containing 3 groups of mopeka_std_values)
        measurements_time = []
        measurements_value = []
        
        # Process 3 groups of measurements
        for group_idx in range(3):
            start_byte = 4 + (group_idx * 5)
            if start_byte + 5 <= len(data):
                # Extract 5 bytes and convert to 40-bit integer
                group_bytes = data[start_byte:start_byte + 5]
                group_value = 0
                for i, byte_val in enumerate(group_bytes):
                    group_value |= (byte_val << (i * 8))
                
                # Extract 4 time/value pairs from the 40-bit value
                for pair_idx in range(4):
                    bit_offset = pair_idx * 10  # Each pair is 10 bits (5 time + 5 value)
                    time_val = (group_value >> bit_offset) & 0x1F  # 5 bits for time
                    value_val = (group_value >> (bit_offset + 5)) & 0x1F  # 5 bits for value
                    
                    # Add 1 to time as per ESPHome implementation
                    measurements_time.append(time_val + 1)
                    measurements_value.append(value_val)
        
        return {
            'device_id': device_id,
            'raw_voltage': raw_voltage,
            'raw_temp': raw_temp,
            'sync_pressed': sync_pressed,
            'slow_update_rate': slow_update_rate,
            'measurements_time': measurements_time,
            'measurements_value': measurements_value
        }

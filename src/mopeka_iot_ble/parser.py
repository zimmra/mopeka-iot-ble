"""Parser for Gmopeka_iot BLE advertisements.

Thanks to https://github.com/spbrogan/mopeka_pro_check for
help decoding the advertisements.

MIT License applies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from bluetooth_data_tools import short_address
from bluetooth_sensor_state_data import BluetoothData
from home_assistant_bluetooth import BluetoothServiceInfo
from sensor_state_data import (
    BinarySensorDeviceClass,
    SensorDeviceClass,
    SensorLibrary,
    Units,
)

from .models import MediumType, config_manager

_LOGGER = logging.getLogger(__name__)


# converting sensor value to height
MOPEKA_TANK_LEVEL_COEFFICIENTS = {
    MediumType.PROPANE: (0.573045, -0.002822, -0.00000535),
    MediumType.AIR: (0.153096, 0.000327, -0.000000294),
    MediumType.FRESH_WATER: (0.600592, 0.003124, -0.00001368),
    MediumType.WASTE_WATER: (0.600592, 0.003124, -0.00001368),
    MediumType.LIVE_WELL: (0.600592, 0.003124, -0.00001368),
    MediumType.BLACK_WATER: (0.600592, 0.003124, -0.00001368),
    MediumType.RAW_WATER: (0.600592, 0.003124, -0.00001368),
    MediumType.GASOLINE: (0.7373417462, -0.001978229885, 0.00000202162),
    MediumType.DIESEL: (0.7373417462, -0.001978229885, 0.00000202162),
    MediumType.LNG: (0.7373417462, -0.001978229885, 0.00000202162),
    MediumType.OIL: (0.7373417462, -0.001978229885, 0.00000202162),
    MediumType.HYDRAULIC_OIL: (0.7373417462, -0.001978229885, 0.00000202162),
}

MOPEKA_MANUFACTURER = 89
MOKPEKA_PRO_SERVICE_UUID = "0000fee5-0000-1000-8000-00805f9b34fb"
# Mopeka Standard sensor constants
MOPEKA_STANDARD_SERVICE_UUID = "0000ada0-0000-1000-8000-00805f9b34fb"
MOPEKA_STANDARD_MANUFACTURER = 0x000D
MOPEKA_STANDARD_DATA_LENGTH = 23

# Standard sensor hardware types
STANDARD_HARDWARE_TYPE = 0x02
XL_HARDWARE_TYPE = 0x03
ETRAILER_HARDWARE_TYPE = 0x46


@dataclass
class MopekaDevice:
    model: str
    name: str
    adv_length: int


DEVICE_TYPES = {
    # Pro sensor models (existing)
    0x3: MopekaDevice("M1017", "Pro Check", 10),
    0x4: MopekaDevice("Pro-200", "Pro-200", 10),
    0x5: MopekaDevice("Pro H20", "Pro Check H2O", 10),
    0x6: MopekaDevice("M1017", "Lippert BottleCheck", 10),
    0x8: MopekaDevice("M1015", "Pro Plus", 10),
    0x9: MopekaDevice("M1015", "Pro Plus with Cellular", 10),
    0xA: MopekaDevice("TD40/TD200", "TD40/TD200", 10),
    0xB: MopekaDevice("TD40/TD200", "TD40/TD200 with Cellular", 10),
    0xC: MopekaDevice("M1017", "Pro Check Universal", 10),
    # Standard sensor models
    0x02: MopekaDevice("Standard", "Standard Sensor", 23),
    0x03: MopekaDevice("XL", "XL Sensor", 23),
    0x46: MopekaDevice("E-Trailer", "E-Trailer Sensor", 23),
}


def hex(data: bytes) -> str:
    """Return a string object containing two hexadecimal digits for each byte in the instance."""
    return "b'{}'".format("".join(f"\\x{b:02x}" for b in data))  # noqa: E231


def battery_to_voltage(battery: int) -> float:
    """Convert battery value to voltage"""
    return battery / 32.0


def battery_to_percentage(battery: int) -> float:
    """Convert battery value to percentage."""
    return round(max(0, min(100, (((battery / 32.0) - 2.2) / 0.65) * 100)), 1)


def temp_to_celsius(temp: int) -> int:
    """Convert temperature value to celsius."""
    return temp - 40


def convert_standard_temperature(raw_temp: int) -> float:
    """Convert Standard sensor raw temperature to Celsius.

    Implements the ESPHome formula for Standard sensor temperature conversion:
    - If raw_temp is 0, return -40.0°C (special case)
    - Otherwise: temperature = (raw_temp - 25.0) * 1.776964

    Args:
        raw_temp: Raw temperature value from Standard sensor (0-63)

    Returns:
        Temperature in degrees Celsius
    """
    if raw_temp == 0:
        return -40.0
    return (raw_temp - 25.0) * 1.776964


def convert_standard_battery(raw_voltage: int) -> tuple[float, float]:
    """Convert Standard sensor raw voltage to voltage and percentage.

    Implements the ESPHome formula for Standard sensor battery conversion:
    - voltage = (raw_voltage / 256.0) * 2.0 + 1.5
    - percentage = min(100, max(0, (voltage - 2.2) / 0.65 * 100))

    Args:
        raw_voltage: Raw voltage value from Standard sensor (0-255)

    Returns:
        Tuple of (voltage_volts, percentage) where:
        - voltage_volts: Battery voltage in volts
        - percentage: Battery percentage (0-100)
    """
    voltage = (raw_voltage / 256.0) * 2.0 + 1.5
    percentage = min(100.0, max(0.0, (voltage - 2.2) / 0.65 * 100))
    return voltage, percentage


def tank_level_to_mm(tank_level: int) -> int:
    """Convert tank level value to mm."""
    return tank_level * 10


def tank_level_and_temp_to_mm(
    tank_level: int, temp: int, medium: MediumType = MediumType.PROPANE
) -> int:
    """Get the tank level in mm for a given fluid type."""
    coefs = MOPEKA_TANK_LEVEL_COEFFICIENTS[medium]
    return int(tank_level * (coefs[0] + (coefs[1] * temp) + (coefs[2] * (temp**2))))


def calculate_lpg_speed_of_sound(temp_c: float, propane_butane_mix: float) -> float:
    """Calculate LPG speed of sound based on temperature and mixture ratio.

    Implements the formula from ESPHome mopeka_std_check implementation:
    speed = 1040.71 - 4.87*temp - 137.5*mix - 0.0107*temp² - 1.63*temp*mix

    Args:
        temp_c: Temperature in Celsius (-40°C to +85°C)
        propane_butane_mix: Propane/butane mixture ratio (0.0 to 1.0)
                           0.0 = 100% butane, 1.0 = 100% propane

    Returns:
        Speed of sound in LPG in meters per second

    Raises:
        ValueError: If temperature or mixture ratio is outside valid range
    """
    # Validate temperature range
    if not (-40.0 <= temp_c <= 85.0):
        raise ValueError(f"Temperature {temp_c}°C is outside valid range -40°C to +85°C")

    # Validate mixture ratio range
    if not (0.0 <= propane_butane_mix <= 1.0):
        raise ValueError(f"Mixture ratio {propane_butane_mix} is outside valid range 0.0 to 1.0")

    # Implement ESPHome formula: speed = 1040.71 - 4.87*temp - 137.5*mix - 0.0107*temp² - 1.63*temp*mix
    speed = (1040.71
             - 4.87 * temp_c
             - 137.5 * propane_butane_mix
             - 0.0107 * (temp_c ** 2)
             - 1.63 * temp_c * propane_butane_mix)

    return speed


def calculate_distance_mm(time_value: int, speed_of_sound: float) -> int:
    """Calculate distance in millimeters from ultrasonic time-of-flight measurement.

    Implements the ESPHome formula for converting ultrasonic timing to distance:
    distance_mm = speed_of_sound * time_value / 100.0

    Matches ESPHome mopeka_std_check implementation at line 173:
    uint32_t distance_value = lpg_speed_of_sound * best_time / 100.0f

    The formula works as follows:
    - time_value is in 10-microsecond ticks from the ultrasonic sensor
    - speed_of_sound is in meters per second
    - Division by 100.0 converts from 10μs ticks to seconds, then result is in meters
    - Result is converted to millimeters and returned as integer

    Args:
        time_value: Time measurement in 10-microsecond ticks (must be non-negative)
        speed_of_sound: Speed of sound in the medium in meters per second (must be positive)

    Returns:
        Distance to target in millimeters as integer, or 0 for invalid inputs

    Examples:
        >>> calculate_distance_mm(1000, 343.0)
        3430
        >>> calculate_distance_mm(0, 343.0)
        0
        >>> calculate_distance_mm(500, 343.0)
        1715
    """
    # Validate input parameters with proper type checking
    if not isinstance(time_value, int):
        raise TypeError("time_value must be an integer")
    if not isinstance(speed_of_sound, (int, float)):
        raise TypeError("speed_of_sound must be a number (int or float)")

    # Validate parameter ranges
    if time_value < 0:
        raise ValueError(f"time_value {time_value} must be non-negative")
    if speed_of_sound <= 0:
        raise ValueError(f"speed_of_sound {speed_of_sound} must be positive")

    # Handle zero time case - return 0 distance for zero time
    if time_value == 0:
        return 0

    # Implement the ESPHome formula: distance_mm = speed_of_sound * time_value / 100.0
    # This matches ESPHome line 173: uint32_t distance_value = lpg_speed_of_sound * best_time / 100.0f
    # Converts time_value (10us ticks) to distance in millimeters
    distance_mm = int(speed_of_sound * time_value / 100.0)
    return distance_mm


def calculate_tank_level_percentage(
    distance_mm: int, empty_distance_mm: int, full_distance_mm: int
) -> float | None:
    """Calculate tank fill percentage from distance measurement and configured tank dimensions.

    Implements the ESPHome formula for calculating tank level percentage:
    percentage = 100 * (empty_distance - distance) / (empty_distance - full_distance)

    This matches the ESPHome mopeka_std_check implementation logic where:
    tank_level = ((100.0f / (this->full_mm_ - this->empty_mm_)) * (distance_value - this->empty_mm_))

    The formula works as follows:
    - When sensor is at empty_distance_mm, percentage = 0%
    - When sensor is at full_distance_mm, percentage = 100%
    - Linear interpolation between these points
    - Result is clamped to 0-100% range

    Args:
        distance_mm: Current distance measurement in millimeters
        empty_distance_mm: Distance when tank is empty (top of liquid when full)
        full_distance_mm: Distance when tank is full (bottom/sensor depth when full)

    Returns:
        Tank fill percentage (0.0 to 100.0), or None if calculation is invalid

    Examples:
        >>> calculate_tank_level_percentage(200, 254, 38)  # 20lb tank example
        75.0
        >>> calculate_tank_level_percentage(254, 254, 38)  # Empty tank
        0.0
        >>> calculate_tank_level_percentage(38, 254, 38)   # Full tank
        100.0
    """
    # Validate input parameter types
    if not isinstance(distance_mm, int):
        raise TypeError("distance_mm must be an integer")
    if not isinstance(empty_distance_mm, int):
        raise TypeError("empty_distance_mm must be an integer")
    if not isinstance(full_distance_mm, int):
        raise TypeError("full_distance_mm must be an integer")
    
    # Validate parameter ranges
    if distance_mm < 0:
        raise ValueError(f"distance_mm {distance_mm} must be non-negative")
    if empty_distance_mm < 0:
        raise ValueError(f"empty_distance_mm {empty_distance_mm} must be non-negative")
    if full_distance_mm < 0:
        raise ValueError(f"full_distance_mm {full_distance_mm} must be non-negative")
    
    # Check for realistic distance values (based on ESPHome tank configurations)
    if distance_mm > 1000:
        return None  # Unrealistic distance measurement
    if empty_distance_mm > 1000 or full_distance_mm > 1000:
        return None  # Unrealistic tank configuration
    
    # Handle division by zero case (empty_distance == full_distance)
    if empty_distance_mm == full_distance_mm:
        return None  # Invalid tank configuration
    
    # Calculate percentage using the standard formula for both normal and inverted tanks
    # The formula is always: percentage = 100 * (empty_distance - distance) / (empty_distance - full_distance)
    # This works correctly for both cases:
    # - Normal tank: empty_distance > full_distance (sensor at top, measures down to liquid)
    # - Inverted tank: empty_distance < full_distance (sensor at bottom, liquid above it)
    percentage = 100.0 * (empty_distance_mm - distance_mm) / (empty_distance_mm - full_distance_mm)
    
    # Clamp percentage to valid 0-100% range
    # This matches ESPHome behavior where tank level is bounded
    clamped_percentage = max(0.0, min(100.0, percentage))
    
    return clamped_percentage


def find_strongest_reflection(measurements: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Find the ultrasonic measurement with the strongest reflection signal.

    Analyzes the measurement array to identify the measurement with the highest
    amplitude value. When multiple measurements have equal amplitude, prefers
    the one with the shortest time (closest reflection).

    Args:
        measurements: List of (time, amplitude) measurement tuples from ultrasonic sensor

    Returns:
        The (time, amplitude) tuple with strongest signal, or None if no valid signal found

    Raises:
        ValueError: If measurements is not a list of exactly 12 tuples
        TypeError: If measurement tuples don't contain exactly 2 integer values
    """
    # Validate input parameters
    if not isinstance(measurements, list):
        raise TypeError("measurements must be a list")

    if len(measurements) != 12:
        raise ValueError(f"measurements must contain exactly 12 tuples, got {len(measurements)}")

    # Validate each measurement tuple
    for i, measurement in enumerate(measurements):
        if not isinstance(measurement, tuple) or len(measurement) != 2:
            raise TypeError(f"measurement {i} must be a tuple of exactly 2 values")

        time_val, amplitude_val = measurement
        if not isinstance(time_val, int) or not isinstance(amplitude_val, int):
            raise TypeError(f"measurement {i} must contain integer values (time, amplitude)")

        if time_val < 0 or amplitude_val < 0:
            raise ValueError(f"measurement {i} time and amplitude must be non-negative")

    # Find measurement with maximum amplitude
    best_measurement = None
    max_amplitude = -1

    for measurement in measurements:
        time_val, amplitude_val = measurement

        # Update best measurement if:
        # 1. Higher amplitude found, OR
        # 2. Equal amplitude but shorter time (closer reflection)
        if (amplitude_val > max_amplitude or
            (amplitude_val == max_amplitude and best_measurement and time_val < best_measurement[0])):
            max_amplitude = amplitude_val
            best_measurement = measurement

    # Return None if all amplitudes are zero (no valid reflection detected)
    if max_amplitude == 0:
        return None

    return best_measurement


class MopekaIOTBluetoothDeviceData(BluetoothData):
    """Data for Mopeka IOT BLE sensors."""

    def __init__(self, medium_type: MediumType = MediumType.PROPANE) -> None:
        super().__init__()
        self._medium_type = medium_type

    def parse_standard_advertisement(self, data: bytes) -> dict:
        """Parse 23-byte Standard sensor manufacturer data structure."""
        import struct
        
        # Parse first 4 bytes: device_id (2 bytes), raw_voltage (1 byte), temp_and_flags (1 byte)
        device_id, raw_voltage, temp_and_flags = struct.unpack('<HBB', data[:4])
        
        # Extract flags and temperature from temp_and_flags byte
        sync_pressed = bool(temp_and_flags & 0x80)
        slow_update_rate = bool(temp_and_flags & 0x40)
        raw_temp = temp_and_flags & 0x3F
        
        # Extract measurement data from bytes 4-22 (19 bytes containing 3 groups of mopeka_std_values)
        # Each mopeka_std_values is 40 bits (5 bytes) containing 4 time/value pairs of 5 bits each
        measurements_time = []
        measurements_value = []
        
        # Process 3 groups of measurements (val[0], val[1], val[2])
        for group_idx in range(3):
            # Each group starts at byte 4 + (group_idx * 5) - but we need to handle bit packing
            # For now, extract 5 bytes per group and unpack the 40-bit structure
            start_byte = 4 + (group_idx * 5)
            if start_byte + 5 <= len(data):
                # Extract 5 bytes and convert to 40-bit integer
                group_bytes = data[start_byte:start_byte + 5]
                # Convert bytes to 40-bit value (little-endian)
                group_value = 0
                for i, byte_val in enumerate(group_bytes):
                    group_value |= (byte_val << (i * 8))
                
                # Extract 4 time/value pairs from the 40-bit value
                # Each time and value is 5 bits
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

    def _start_update(self, service_info: BluetoothServiceInfo) -> None:
        """Update from BLE advertisement data."""
        _LOGGER.debug(
            "Parsing Mopeka IOT BLE advertisement data: %s, MediumType is: %s",
            service_info,
            self._medium_type,
        )
        manufacturer_data = service_info.manufacturer_data
        service_uuids = service_info.service_uuids
        address = service_info.address
        if (
            MOPEKA_MANUFACTURER not in manufacturer_data
            or (MOKPEKA_PRO_SERVICE_UUID not in service_uuids
                and MOPEKA_STANDARD_SERVICE_UUID not in service_uuids)
        ):
            _LOGGER.debug("Not a Mopeka IOT BLE advertisement: %s", service_info)
            return

        # Route to Standard or Pro sensor parsing
        if MOPEKA_STANDARD_SERVICE_UUID in service_uuids:
            # Standard sensor validation and parsing
            if (MOPEKA_STANDARD_MANUFACTURER not in manufacturer_data
                or len(manufacturer_data[MOPEKA_STANDARD_MANUFACTURER]) != MOPEKA_STANDARD_DATA_LENGTH):
                _LOGGER.debug("Invalid Standard sensor data: %s", service_info)
                return
            
            # Parse Standard sensor data
            data = manufacturer_data[MOPEKA_STANDARD_MANUFACTURER]
            parsed_data = self.parse_standard_advertisement(data)
            
            # Enhanced device registration for Standard sensors with Home Assistant compatibility
            device_config = config_manager.get_device_config(address)
            
            # Set comprehensive device information for Home Assistant
            self.set_device_manufacturer("Mopeka")
            
            # Determine sensor hardware type from data if possible
            sensor_hw_type = "Standard"
            if device_config and device_config.sensor_type in ["XL", "ETRAILER"]:
                sensor_hw_type = device_config.sensor_type
            
            # Set device type with hardware variant
            self.set_device_type(f"Mopeka {sensor_hw_type} Tank Level Sensor")
            
            # Set device name with configuration-aware naming
            if device_config and device_config.device_name:
                device_name = device_config.device_name
            else:
                device_name = f"Mopeka {sensor_hw_type} {short_address(address)}"
            self.set_device_name(device_name)
            
            # Set firmware version for device registry (Standard sensors)
            self.set_device_sw_version("Standard v1.0")

            # Set hardware version if configuration provides sensor type details
            if device_config and device_config.sensor_type != "STANDARD":
                self.set_device_hw_version(f"{device_config.sensor_type} Hardware")
            
            # Convert raw values using Standard-specific formulas
            battery_voltage = convert_standard_battery(parsed_data['raw_voltage'])[0]
            battery_percentage = convert_standard_battery(parsed_data['raw_voltage'])[1]
            temp_celsius = convert_standard_temperature(parsed_data['raw_temp'])
            
            # Update sensors with enhanced Home Assistant metadata and unique IDs
            # Temperature sensor with proper device class and unique ID
            self.update_predefined_sensor(
                SensorLibrary.TEMPERATURE__CELSIUS, 
                temp_celsius,
                key=f"{address}_temperature",
                name="Temperature",
            )
            
            # Battery percentage with battery device class and unique ID
            self.update_predefined_sensor(
                SensorLibrary.BATTERY__PERCENTAGE, 
                battery_percentage,
                key=f"{address}_battery_level",
                name="Battery Level",
            )
            
            # Battery voltage sensor with voltage device class and unique ID
            self.update_predefined_sensor(
                SensorLibrary.VOLTAGE__ELECTRIC_POTENTIAL_VOLT,
                battery_voltage,
                name="Battery Voltage", 
                key=f"{address}_battery_voltage",
            )
            
            # Sync button pressed status with occupancy device class and unique ID
            self.update_predefined_binary_sensor(
                BinarySensorDeviceClass.OCCUPANCY,
                parsed_data['sync_pressed'],
                key=f"{address}_sync_pressed",
                name="Sync Button Pressed",
            )
            
            # Slow update rate status with running device class and unique ID
            self.update_predefined_binary_sensor(
                BinarySensorDeviceClass.RUNNING,
                parsed_data['slow_update_rate'],
                key=f"{address}_slow_update_rate", 
                name="Slow Update Mode",
            )
            
            # Calculate tank level for Standard sensors with configuration support
            tank_level_mm = None
            tank_level_percentage = None

            if parsed_data['measurements_time'] and parsed_data['measurements_value']:
                # Create measurement tuples for strongest reflection analysis
                measurements = list(zip(
                    parsed_data['measurements_time'],
                    parsed_data['measurements_value']
                ))

                # Find strongest reflection signal
                strongest = find_strongest_reflection(measurements)
                if strongest:
                    time_value, amplitude = strongest

                    # Calculate distance using configured or default mixture ratio
                    if self._medium_type == MediumType.PROPANE:
                        # Use configured mixture ratio if available, otherwise default
                        propane_butane_mix = device_config.propane_butane_mix if device_config else 0.8
                        speed_of_sound = calculate_lpg_speed_of_sound(temp_celsius, propane_butane_mix)
                    else:
                        # For non-LPG mediums, use simplified sound speed (air speed as fallback)
                        speed_of_sound = 343.0  # Speed of sound in air at 20°C

                    tank_level_mm = calculate_distance_mm(time_value, speed_of_sound)

                    # Calculate tank level percentage if device configuration available
                    if device_config:
                        tank_level_percentage = calculate_tank_level_percentage(
                            tank_level_mm,
                            device_config.empty_distance_mm,
                            device_config.full_distance_mm
                        )
                        _LOGGER.debug(
                            "Tank level calculated: %smm (%.1f%%) using config %s",
                            tank_level_mm, tank_level_percentage or 0, device_config.tank_description
                        )
                    
            # Tank level distance sensor with proper Home Assistant device class and unique ID
            self.update_sensor(
                f"{address}_tank_level",
                Units.LENGTH_MILLIMETERS,
                tank_level_mm,
                SensorDeviceClass.DISTANCE,
                "Tank Level Distance",
            )
            
            # Tank level percentage sensor with unique ID (requires configuration)
            self.update_sensor(
                f"{address}_tank_level_percentage",
                Units.PERCENTAGE,
                tank_level_percentage,
                None,
                "Tank Level",
            )
            
            # Ultrasonic measurement quality sensor with unique ID
            if parsed_data['measurements_value']:
                max_amplitude = max(parsed_data['measurements_value'])
                quality_percentage = min(100, max_amplitude * 100 // 31)  # Scale 0-31 to 0-100%
                self.update_sensor(
                    f"{address}_reading_quality",
                    Units.PERCENTAGE,
                    quality_percentage,
                    None,
                    "Signal Quality",
                )
            
        else:
            # Pro sensor parsing with enhanced Home Assistant device registration
            data = manufacturer_data[MOPEKA_MANUFACTURER]
            model_num = data[0]
            if not (device_type := DEVICE_TYPES.get(model_num)):
                _LOGGER.debug("Unsupported Mopeka IOT BLE advertisement: %s", service_info)
                return
            adv_length = device_type.adv_length
            if len(data) != adv_length:
                return

            # Enhanced device registration for Pro sensors with Home Assistant compatibility
            self.set_device_manufacturer("Mopeka")
            self.set_device_type(f"Mopeka {device_type.model} Tank Level Sensor")
            self.set_device_name(f"Mopeka {device_type.name} {short_address(address)}")
            
            # Set firmware and hardware versions for device registry (Pro sensors)
            firmware_version = getattr(device_type, 'firmware_version', f"Pro v{device_type.model}")
            self.set_device_sw_version(firmware_version)
            self.set_device_hw_version(f"{device_type.model} Hardware")
            
            # Parse Pro sensor data
            battery = data[1]
            battery_voltage = battery_to_voltage(battery)
            battery_percentage = battery_to_percentage(battery)
            button_pressed = bool(data[2] & 0x80 > 0)
            temp = data[2] & 0x7F
            temp_celsius = temp_to_celsius(temp)
            tank_level = ((int(data[4]) << 8) + data[3]) & 0x3FFF
            tank_level_mm = tank_level_and_temp_to_mm(tank_level, temp, self._medium_type)
            reading_quality = data[4] >> 6
            accelerometer_x = data[8]
            accelerometer_y = data[9]

            # Update Pro sensors with proper Home Assistant device classes and unique IDs
            self.update_predefined_sensor(
                SensorLibrary.TEMPERATURE__CELSIUS, 
                temp_celsius,
                key=f"{address}_temperature",
                name="Temperature"
            )
            self.update_predefined_sensor(
                SensorLibrary.BATTERY__PERCENTAGE, 
                battery_percentage,
                key=f"{address}_battery_level",
                name="Battery Level"
            )
            self.update_predefined_sensor(
                SensorLibrary.VOLTAGE__ELECTRIC_POTENTIAL_VOLT,
                battery_voltage,
                name="Battery Voltage",
                key=f"{address}_battery_voltage",
            )
            self.update_predefined_binary_sensor(
                BinarySensorDeviceClass.OCCUPANCY,
                button_pressed,
                key=f"{address}_button_pressed",
                name="Button Pressed",
            )
            self.update_sensor(
                f"{address}_tank_level",
                Units.LENGTH_MILLIMETERS,
                tank_level_mm if reading_quality >= 1 else None,
                SensorDeviceClass.DISTANCE,
                "Tank Level",
            )
            self.update_sensor(
                f"{address}_accelerometer_x",
                None,
                accelerometer_x,
                None,
                "Position X",
            )
            self.update_sensor(
                f"{address}_accelerometer_y",
                None,
                accelerometer_y,
                None,
                "Position Y",
            )
            self.update_sensor(
                f"{address}_reading_quality_raw",
                None,
                reading_quality,
                None,
                "Signal Quality Raw",
            )
            self.update_sensor(
                f"{address}_reading_quality",
                Units.PERCENTAGE,
                round(reading_quality / 3 * 100),
                None,
                "Signal Quality",
            )
        # Reading stars = (3-reading_quality) * "★" + (reading_quality * "⭐")

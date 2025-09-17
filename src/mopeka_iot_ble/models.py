"""Types for Mopeka IOT BLE advertisements.


Thanks to https://github.com/spbrogan/mopeka_pro_check for
help decoding the advertisements.

MIT License applies.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal


class MediumType(Enum):
    """Enumeration of medium types for tank level measurements."""

    PROPANE = "propane"
    AIR = "air"
    FRESH_WATER = "fresh_water"
    WASTE_WATER = "waste_water"
    LIVE_WELL = "live_well"
    BLACK_WATER = "black_water"
    RAW_WATER = "raw_water"
    GASOLINE = "gasoline"
    DIESEL = "diesel"
    LNG = "lng"
    OIL = "oil"
    HYDRAULIC_OIL = "hydraulic_oil"


@dataclass 
class TankPreset:
    """Tank geometry preset configuration.
    
    Based on ESPHome mopeka_std_check tank presets for common tank types.
    """
    name: str
    empty_distance_mm: int
    full_distance_mm: int
    description: str


# Standard tank presets based on ESPHome mopeka_std_check reference
TANK_PRESETS = {
    "NORTH_AMERICA_20LB_VERTICAL": TankPreset(
        name="North America 20lb Vertical",
        empty_distance_mm=381,  # 15 inches
        full_distance_mm=51,    # 2 inches  
        description="Standard 20lb propane tank (vertical mount)"
    ),
    "NORTH_AMERICA_30LB_VERTICAL": TankPreset(
        name="North America 30lb Vertical", 
        empty_distance_mm=508,  # 20 inches
        full_distance_mm=51,    # 2 inches
        description="Standard 30lb propane tank (vertical mount)"
    ),
    "NORTH_AMERICA_40LB_VERTICAL": TankPreset(
        name="North America 40lb Vertical",
        empty_distance_mm=635,  # 25 inches  
        full_distance_mm=51,    # 2 inches
        description="Standard 40lb propane tank (vertical mount)"
    ),
    "EUROPE_11KG": TankPreset(
        name="Europe 11kg",
        empty_distance_mm=381,  # 15 inches
        full_distance_mm=51,    # 2 inches  
        description="European 11kg propane tank"
    ),
    "EUROPE_13KG": TankPreset(
        name="Europe 13kg", 
        empty_distance_mm=432,  # 17 inches
        full_distance_mm=51,    # 2 inches
        description="European 13kg propane tank"
    ),
    "CUSTOM": TankPreset(
        name="Custom",
        empty_distance_mm=0,
        full_distance_mm=0, 
        description="Custom tank dimensions"
    )
}


@dataclass
class MopekaDeviceConfig:
    """Complete configuration for a Mopeka Standard device.
    
    Includes tank geometry, mixture settings, and device identification.
    Supports both preset tank configurations and custom dimensions.
    """
    
    mac_address: str
    tank_preset: str = "CUSTOM"
    custom_empty_distance_mm: int | None = None
    custom_full_distance_mm: int | None = None
    propane_butane_mix: float = 0.8
    sensor_type: Literal["STANDARD", "XL", "ETRAILER"] = "STANDARD"
    device_name: str | None = None
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Validate MAC address format
        if not self.mac_address:
            raise ValueError("mac_address cannot be empty")
            
        # Validate tank preset exists
        if self.tank_preset not in TANK_PRESETS:
            raise ValueError(f"tank_preset must be one of: {list(TANK_PRESETS.keys())}")
            
        # Validate propane/butane mixture ratio
        if not 0.0 <= self.propane_butane_mix <= 1.0:
            raise ValueError("propane_butane_mix must be between 0.0 and 1.0")
            
        # Validate sensor type
        if self.sensor_type not in ["STANDARD", "XL", "ETRAILER"]:
            raise ValueError("sensor_type must be STANDARD, XL, or ETRAILER")
            
        # For custom tanks, validate distance parameters
        if self.tank_preset == "CUSTOM":
            if self.custom_empty_distance_mm is None or self.custom_full_distance_mm is None:
                raise ValueError("custom_empty_distance_mm and custom_full_distance_mm required for CUSTOM preset")
                
            if self.custom_empty_distance_mm <= 0:
                raise ValueError("custom_empty_distance_mm must be positive")
            if self.custom_full_distance_mm <= 0:
                raise ValueError("custom_full_distance_mm must be positive")
                
            # For tank level sensors, empty distance should be greater than full distance
            # (sensor measures down from top of tank to liquid surface)
            if self.custom_empty_distance_mm <= self.custom_full_distance_mm:
                raise ValueError("custom_empty_distance_mm must be greater than custom_full_distance_mm")
    
    @property
    def empty_distance_mm(self) -> int:
        """Get effective empty distance based on preset or custom setting."""
        if self.tank_preset == "CUSTOM":
            return self.custom_empty_distance_mm
        return TANK_PRESETS[self.tank_preset].empty_distance_mm
    
    @property  
    def full_distance_mm(self) -> int:
        """Get effective full distance based on preset or custom setting."""
        if self.tank_preset == "CUSTOM":
            return self.custom_full_distance_mm
        return TANK_PRESETS[self.tank_preset].full_distance_mm
        
    @property
    def tank_description(self) -> str:
        """Get tank description based on preset or custom setting."""
        if self.tank_preset == "CUSTOM":
            return f"Custom tank ({self.custom_empty_distance_mm}mm - {self.custom_full_distance_mm}mm)"
        return TANK_PRESETS[self.tank_preset].description
    
    def to_standard_device(self) -> "MopekaStandardDevice":
        """Convert to legacy MopekaStandardDevice format for compatibility."""
        return MopekaStandardDevice(
            mac_address=self.mac_address,
            empty_distance_mm=self.empty_distance_mm,
            full_distance_mm=self.full_distance_mm,
            propane_butane_mix=self.propane_butane_mix,
            sensor_type=self.sensor_type
        )


class ConfigurationManager:
    """Manager for storing and retrieving Mopeka device configurations.
    
    Provides a simple interface for configuration persistence that can be
    adapted for different storage backends (file system, database, etc.).
    """
    
    def __init__(self) -> None:
        """Initialize configuration manager with in-memory storage."""
        self._configs: Dict[str, MopekaDeviceConfig] = {}
    
    def add_device_config(self, config: MopekaDeviceConfig) -> None:
        """Add or update device configuration.
        
        Args:
            config: MopekaDeviceConfig instance to store
        """
        self._configs[config.mac_address.upper()] = config
    
    def get_device_config(self, mac_address: str) -> MopekaDeviceConfig | None:
        """Get device configuration by MAC address.
        
        Args:
            mac_address: Device MAC address (case-insensitive)
            
        Returns:
            MopekaDeviceConfig instance or None if not found
        """
        return self._configs.get(mac_address.upper())
    
    def remove_device_config(self, mac_address: str) -> bool:
        """Remove device configuration.
        
        Args:
            mac_address: Device MAC address (case-insensitive)
            
        Returns:
            True if configuration was removed, False if not found
        """
        return self._configs.pop(mac_address.upper(), None) is not None
    
    def list_configured_devices(self) -> List[str]:
        """Get list of configured device MAC addresses.
        
        Returns:
            List of MAC addresses that have configurations
        """
        return list(self._configs.keys())
    
    def has_device_config(self, mac_address: str) -> bool:
        """Check if device has configuration.
        
        Args:
            mac_address: Device MAC address (case-insensitive)
            
        Returns:
            True if device has configuration
        """
        return mac_address.upper() in self._configs
    
    def get_all_configs(self) -> Dict[str, MopekaDeviceConfig]:
        """Get all device configurations.
        
        Returns:
            Dictionary mapping MAC addresses to configurations
        """
        return self._configs.copy()
    
    def clear_all_configs(self) -> None:
        """Remove all device configurations."""
        self._configs.clear()


# Global configuration manager instance
config_manager = ConfigurationManager()


def create_device_config_from_preset(
    mac_address: str,
    tank_preset: str,
    propane_butane_mix: float = 0.8,
    sensor_type: Literal["STANDARD", "XL", "ETRAILER"] = "STANDARD",
    device_name: str | None = None
) -> MopekaDeviceConfig:
    """Create device configuration from tank preset.
    
    Args:
        mac_address: Device MAC address
        tank_preset: Tank preset name from TANK_PRESETS
        propane_butane_mix: Propane/butane mixture ratio (0.0 to 1.0)
        sensor_type: Sensor hardware type
        device_name: Optional custom device name
        
    Returns:
        MopekaDeviceConfig instance
        
    Raises:
        ValueError: If preset name is invalid
    """
    return MopekaDeviceConfig(
        mac_address=mac_address,
        tank_preset=tank_preset,
        propane_butane_mix=propane_butane_mix,
        sensor_type=sensor_type,
        device_name=device_name
    )


def create_custom_device_config(
    mac_address: str,
    empty_distance_mm: int,
    full_distance_mm: int,
    propane_butane_mix: float = 0.8,
    sensor_type: Literal["STANDARD", "XL", "ETRAILER"] = "STANDARD", 
    device_name: str | None = None
) -> MopekaDeviceConfig:
    """Create device configuration with custom tank dimensions.
    
    Args:
        mac_address: Device MAC address
        empty_distance_mm: Distance when tank is empty (millimeters)
        full_distance_mm: Distance when tank is full (millimeters)
        propane_butane_mix: Propane/butane mixture ratio (0.0 to 1.0)
        sensor_type: Sensor hardware type
        device_name: Optional custom device name
        
    Returns:
        MopekaDeviceConfig instance
        
    Raises:
        ValueError: If parameters are invalid
    """
    return MopekaDeviceConfig(
        mac_address=mac_address,
        tank_preset="CUSTOM",
        custom_empty_distance_mm=empty_distance_mm,
        custom_full_distance_mm=full_distance_mm,
        propane_butane_mix=propane_butane_mix,
        sensor_type=sensor_type,
        device_name=device_name
    )


@dataclass
class MopekaStandardDevice:
    """Configuration for Mopeka Standard sensor devices.
    
    Based on ESPHome mopeka_std_check implementation configuration fields.
    """
    
    mac_address: str
    empty_distance_mm: int
    full_distance_mm: int
    propane_butane_mix: float = 0.8
    sensor_type: Literal["STANDARD", "XL", "ETRAILER"] = "STANDARD"
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.empty_distance_mm <= 0:
            raise ValueError("empty_distance_mm must be positive")
        if self.full_distance_mm <= 0:
            raise ValueError("full_distance_mm must be positive")
        # For tank level sensors, empty distance should be greater than full distance
        # (sensor measures down from top of tank to liquid surface)
        if self.empty_distance_mm <= self.full_distance_mm:
            raise ValueError("empty_distance_mm must be greater than full_distance_mm")
        if not 0.0 <= self.propane_butane_mix <= 1.0:
            raise ValueError("propane_butane_mix must be between 0.0 and 1.0")
        if self.sensor_type not in ["STANDARD", "XL", "ETRAILER"]:
            raise ValueError("sensor_type must be STANDARD, XL, or ETRAILER")


@dataclass
class SensorReading:
    """Sensor reading data from a Mopeka Standard sensor.
    
    Contains all measurement data from a Standard sensor reading,
    including tank level, environmental data, and device status.
    """
    
    mac_address: str
    tank_level_mm: int | None  # Distance to liquid surface in millimeters
    tank_level_percentage: float | None  # Tank fill percentage (0-100%)
    temperature_celsius: float  # Temperature in Celsius
    battery_percentage: float  # Battery percentage (0-100%)
    battery_voltage: float  # Battery voltage in volts
    sync_pressed: bool  # Whether sync button was pressed
    slow_update_rate: bool  # Whether device is in slow update mode
    timestamp: datetime  # When reading was taken
    signal_strength: int | None = None  # RSSI signal strength in dBm
    
    def __post_init__(self) -> None:
        """Validate sensor reading data."""
        if not self.mac_address:
            raise ValueError("mac_address cannot be empty")
        if self.tank_level_mm is not None and self.tank_level_mm < 0:
            raise ValueError("tank_level_mm must be non-negative")
        if self.tank_level_percentage is not None and not (0.0 <= self.tank_level_percentage <= 100.0):
            raise ValueError("tank_level_percentage must be between 0.0 and 100.0")
        if not isinstance(self.battery_percentage, (int, float)) or not (0.0 <= self.battery_percentage <= 100.0):
            raise ValueError("battery_percentage must be between 0.0 and 100.0")
        if not isinstance(self.battery_voltage, (int, float)) or self.battery_voltage < 0:
            raise ValueError("battery_voltage must be non-negative")

"""
ExternalDataConnector - Integration with external data sources for Time Series Pro
Part of Epic #2: Advanced Data Science Features
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import json
import warnings
warnings.filterwarnings('ignore')

# External data libraries
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False

from models import ExternalDataSource
from app import db


class BaseDataConnector:
    """Base class for all external data connectors"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    def fetch(self, *args, **kwargs) -> pd.DataFrame:
        """Fetch data from external source"""
        raise NotImplementedError("Subclasses must implement fetch method")
    
    def validate_config(self) -> bool:
        """Validate connector configuration"""
        raise NotImplementedError("Subclasses must implement validate_config method")
    
    def get_available_fields(self) -> List[str]:
        """Get list of available data fields"""
        raise NotImplementedError("Subclasses must implement get_available_fields method")


class HolidayConnector(BaseDataConnector):
    """Connector for holiday data using the holidays library"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.supported_countries = [
            'US', 'CA', 'UK', 'DE', 'FR', 'JP', 'AU', 'IN', 'BR', 'MX', 
            'IT', 'ES', 'NL', 'SE', 'NO', 'DK', 'FI', 'CH', 'AT', 'BE'
        ] if HOLIDAYS_AVAILABLE else []
    
    def fetch(self, country_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch holiday data for date range"""
        if not HOLIDAYS_AVAILABLE:
            raise ImportError("holidays library not available")
        
        if country_code not in self.supported_countries:
            raise ValueError(f"Country code {country_code} not supported. Available: {self.supported_countries}")
        
        try:
            # Convert date strings to datetime objects
            start_dt = pd.to_datetime(start_date).date()
            end_dt = pd.to_datetime(end_date).date()
            
            # Get holidays for the country
            country_holidays = holidays.country_holidays(country_code)
            
            # Create date range
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
            
            # Create DataFrame with holiday information
            holiday_data = []
            for date in date_range:
                date_obj = date.date()
                is_holiday = date_obj in country_holidays
                holiday_name = country_holidays.get(date_obj, '') if is_holiday else ''
                
                holiday_data.append({
                    'date': date,
                    'is_holiday': is_holiday,
                    'holiday_name': holiday_name,
                    'is_weekend': date.dayofweek >= 5,
                    'country_code': country_code
                })
            
            df = pd.DataFrame(holiday_data)
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to fetch holiday data: {str(e)}")
    
    def validate_config(self) -> bool:
        """Validate holiday connector configuration"""
        return HOLIDAYS_AVAILABLE and len(self.supported_countries) > 0
    
    def get_available_fields(self) -> List[str]:
        return ['date', 'is_holiday', 'holiday_name', 'is_weekend', 'country_code']
    
    def get_supported_countries(self) -> List[str]:
        return self.supported_countries


class WeatherAPIConnector(BaseDataConnector):
    """Connector for weather data from OpenWeatherMap or similar APIs"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = self.config.get('api_key')
        self.base_url = self.config.get('base_url', 'https://api.openweathermap.org/data/2.5')
    
    def fetch(self, location: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch weather data for location and date range"""
        if not self.api_key:
            raise ValueError("Weather API key not configured")
        
        try:
            # Note: This is a simplified implementation
            # In practice, you'd need to handle historical data APIs
            # which often require different endpoints and may have costs
            
            # For demonstration, create mock weather data
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
            
            weather_data = []
            base_temp = 20  # Base temperature in Celsius
            
            for i, date in enumerate(date_range):
                # Generate realistic-looking mock weather data
                seasonal_temp = base_temp + 15 * np.sin(2 * np.pi * date.dayofyear / 365.25)
                daily_variation = np.random.normal(0, 5)
                temperature = seasonal_temp + daily_variation
                
                humidity = max(0, min(100, np.random.normal(60, 15)))
                precipitation = max(0, np.random.exponential(2) if np.random.random() < 0.3 else 0)
                wind_speed = max(0, np.random.normal(10, 5))
                pressure = np.random.normal(1013.25, 10)
                
                weather_data.append({
                    'date': date,
                    'location': location,
                    'temperature_celsius': round(temperature, 1),
                    'temperature_fahrenheit': round(temperature * 9/5 + 32, 1),
                    'humidity_percent': round(humidity, 1),
                    'precipitation_mm': round(precipitation, 1),
                    'wind_speed_kmh': round(wind_speed, 1),
                    'pressure_hpa': round(pressure, 1),
                    'weather_condition': self._get_weather_condition(temperature, precipitation, humidity)
                })
            
            df = pd.DataFrame(weather_data)
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to fetch weather data: {str(e)}")
    
    def _get_weather_condition(self, temp: float, precip: float, humidity: float) -> str:
        """Determine weather condition based on parameters"""
        if precip > 10:
            return 'heavy_rain'
        elif precip > 2:
            return 'rain'
        elif precip > 0.1:
            return 'light_rain'
        elif humidity > 80:
            return 'cloudy'
        elif temp > 25:
            return 'sunny'
        elif temp < 5:
            return 'cold'
        else:
            return 'clear'
    
    def fetch_current_weather(self, location: str) -> Dict:
        """Fetch current weather data"""
        if not self.api_key:
            raise ValueError("Weather API key not configured")
        
        try:
            # This would be a real API call in practice
            url = f"{self.base_url}/weather"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            # For demo, return mock data
            return {
                'location': location,
                'temperature': 22.5,
                'humidity': 65,
                'pressure': 1013,
                'wind_speed': 12,
                'description': 'partly cloudy',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Failed to fetch current weather: {str(e)}")
    
    def validate_config(self) -> bool:
        """Validate weather connector configuration"""
        return bool(self.api_key)
    
    def get_available_fields(self) -> List[str]:
        return [
            'date', 'location', 'temperature_celsius', 'temperature_fahrenheit',
            'humidity_percent', 'precipitation_mm', 'wind_speed_kmh', 
            'pressure_hpa', 'weather_condition'
        ]


class EconomicDataConnector(BaseDataConnector):
    """Connector for economic indicators (GDP, inflation, etc.)"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = self.config.get('api_key')
        self.data_source = self.config.get('source', 'fred')  # Federal Reserve Economic Data
    
    def fetch(self, indicator: str, start_date: str, end_date: str, country: str = 'US') -> pd.DataFrame:
        """Fetch economic indicator data"""
        try:
            # Generate mock economic data for demonstration
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Use monthly frequency for economic data
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='M')
            
            economic_data = []
            base_value = self._get_base_value(indicator)
            
            for i, date in enumerate(date_range):
                # Generate realistic economic data trends
                trend = base_value * (1 + 0.02 * i / 12)  # 2% annual trend
                seasonal = 0.05 * np.sin(2 * np.pi * date.month / 12)  # Seasonal component
                noise = np.random.normal(0, 0.02)  # Random noise
                
                value = trend * (1 + seasonal + noise)
                
                economic_data.append({
                    'date': date,
                    'indicator': indicator,
                    'value': round(value, 4),
                    'country': country,
                    'source': self.data_source,
                    'units': self._get_indicator_units(indicator)
                })
            
            df = pd.DataFrame(economic_data)
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to fetch economic data: {str(e)}")
    
    def _get_base_value(self, indicator: str) -> float:
        """Get base value for economic indicator"""
        base_values = {
            'gdp': 20000,  # GDP in billions
            'inflation': 2.5,  # Inflation rate percentage
            'unemployment': 5.0,  # Unemployment rate percentage
            'interest_rate': 2.0,  # Interest rate percentage
            'stock_index': 3000,  # Stock market index
            'exchange_rate': 1.0,  # Exchange rate
            'consumer_confidence': 100,  # Consumer confidence index
            'housing_starts': 1.2  # Housing starts in millions
        }
        return base_values.get(indicator, 100)
    
    def _get_indicator_units(self, indicator: str) -> str:
        """Get units for economic indicator"""
        units = {
            'gdp': 'billions_usd',
            'inflation': 'percent',
            'unemployment': 'percent',
            'interest_rate': 'percent',
            'stock_index': 'index',
            'exchange_rate': 'ratio',
            'consumer_confidence': 'index',
            'housing_starts': 'millions'
        }
        return units.get(indicator, 'units')
    
    def get_available_indicators(self) -> List[str]:
        """Get list of available economic indicators"""
        return [
            'gdp', 'inflation', 'unemployment', 'interest_rate',
            'stock_index', 'exchange_rate', 'consumer_confidence', 'housing_starts'
        ]
    
    def validate_config(self) -> bool:
        """Validate economic data connector configuration"""
        return True  # Mock data doesn't require API key
    
    def get_available_fields(self) -> List[str]:
        return ['date', 'indicator', 'value', 'country', 'source', 'units']


class EventAPIConnector(BaseDataConnector):
    """Connector for event data (sports, concerts, conferences, etc.)"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = self.config.get('api_key')
        self.event_types = self.config.get('event_types', ['sports', 'concerts', 'conferences'])
    
    def fetch(self, location: str, start_date: str, end_date: str, event_type: str = None) -> pd.DataFrame:
        """Fetch event data for location and date range"""
        try:
            # Generate mock event data
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            events = []
            current_date = start_dt
            
            while current_date <= end_dt:
                # Random chance of events on each day
                if np.random.random() < 0.1:  # 10% chance of event
                    event_types = [event_type] if event_type else self.event_types
                    selected_type = np.random.choice(event_types)
                    
                    # Generate event based on type
                    event_data = self._generate_event(current_date, location, selected_type)
                    events.append(event_data)
                
                current_date += timedelta(days=1)
            
            df = pd.DataFrame(events)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df['event_impact'] = self._calculate_event_impact(df)
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to fetch event data: {str(e)}")
    
    def _generate_event(self, date: datetime, location: str, event_type: str) -> Dict:
        """Generate mock event data"""
        event_names = {
            'sports': ['Football Game', 'Basketball Game', 'Baseball Game', 'Soccer Match'],
            'concerts': ['Rock Concert', 'Pop Concert', 'Jazz Festival', 'Classical Concert'],
            'conferences': ['Tech Conference', 'Business Summit', 'Academic Conference', 'Trade Show']
        }
        
        capacities = {
            'sports': np.random.randint(20000, 80000),
            'concerts': np.random.randint(5000, 50000),
            'conferences': np.random.randint(500, 10000)
        }
        
        return {
            'date': date,
            'location': location,
            'event_type': event_type,
            'event_name': np.random.choice(event_names[event_type]),
            'estimated_attendance': capacities[event_type],
            'venue': f"{location} {event_type.title()} Venue",
            'duration_hours': np.random.randint(2, 8),
            'is_weekend': date.dayofweek >= 5
        }
    
    def _calculate_event_impact(self, df: pd.DataFrame) -> List[float]:
        """Calculate potential impact score for events"""
        impact_scores = []
        for _, row in df.iterrows():
            base_impact = {
                'sports': 0.8,
                'concerts': 0.6,
                'conferences': 0.4
            }.get(row['event_type'], 0.3)
            
            # Adjust for attendance
            attendance_factor = min(row['estimated_attendance'] / 50000, 1.0)
            
            # Adjust for weekend
            weekend_factor = 1.2 if row['is_weekend'] else 1.0
            
            impact = base_impact * attendance_factor * weekend_factor
            impact_scores.append(round(impact, 2))
        
        return impact_scores
    
    def validate_config(self) -> bool:
        """Validate event connector configuration"""
        return True  # Mock data doesn't require API key
    
    def get_available_fields(self) -> List[str]:
        return [
            'date', 'location', 'event_type', 'event_name', 'estimated_attendance',
            'venue', 'duration_hours', 'is_weekend', 'event_impact'
        ]


class ExternalDataConnector:
    """Main orchestrator for external data integration"""
    
    def __init__(self, project_id: int = None):
        self.project_id = project_id
        self.connectors = {}
        self._initialize_connectors()
    
    def _initialize_connectors(self):
        """Initialize all available connectors"""
        self.connectors = {
            'holidays': HolidayConnector(),
            'weather': WeatherAPIConnector(),
            'economic': EconomicDataConnector(),
            'events': EventAPIConnector()
        }
    
    def get_available_sources(self) -> Dict[str, List[str]]:
        """Get available external data sources and their fields"""
        available = {}
        
        for source_name, connector in self.connectors.items():
            if connector.validate_config():
                available[source_name] = {
                    'fields': connector.get_available_fields(),
                    'status': 'available'
                }
            else:
                available[source_name] = {
                    'fields': connector.get_available_fields(),
                    'status': 'configuration_required'
                }
        
        return available
    
    def fetch_holiday_data(self, country_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch holiday data for date range"""
        return self.connectors['holidays'].fetch(country_code, start_date, end_date)
    
    def fetch_weather_data(self, location: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
        """Fetch weather data from APIs"""
        if api_key:
            self.connectors['weather'].config['api_key'] = api_key
        
        return self.connectors['weather'].fetch(location, start_date, end_date)
    
    def fetch_economic_data(self, indicator: str, start_date: str, end_date: str, country: str = 'US') -> pd.DataFrame:
        """Fetch economic indicator data"""
        return self.connectors['economic'].fetch(indicator, start_date, end_date, country)
    
    def fetch_event_data(self, location: str, start_date: str, end_date: str, event_type: str = None) -> pd.DataFrame:
        """Fetch event data for location and date range"""
        return self.connectors['events'].fetch(location, start_date, end_date, event_type)
    
    def configure_data_source(self, source_type: str, configuration: Dict, 
                            data_mapping: Dict = None, is_active: bool = True):
        """Configure external data source for project"""
        if not self.project_id:
            raise ValueError("Project ID required for data source configuration")
        
        try:
            # Check if data source already exists
            existing_source = ExternalDataSource.query.filter_by(
                project_id=self.project_id,
                source_type=source_type
            ).first()
            
            if existing_source:
                existing_source.set_api_configuration(configuration)
                existing_source.set_data_mapping(data_mapping or {})
                existing_source.is_active = is_active
            else:
                data_source = ExternalDataSource(
                    project_id=self.project_id,
                    source_type=source_type,
                    is_active=is_active
                )
                data_source.set_api_configuration(configuration)
                data_source.set_data_mapping(data_mapping or {})
                
                db.session.add(data_source)
            
            db.session.commit()
            
        except Exception as e:
            db.session.rollback()
            raise Exception(f"Failed to configure data source: {str(e)}")
    
    def get_configured_sources(self) -> List[Dict]:
        """Get configured external data sources for project"""
        if not self.project_id:
            return []
        
        try:
            sources = ExternalDataSource.query.filter_by(project_id=self.project_id).all()
            
            configured_sources = []
            for source in sources:
                configured_sources.append({
                    'id': source.id,
                    'source_type': source.source_type,
                    'configuration': source.get_api_configuration(),
                    'data_mapping': source.get_data_mapping(),
                    'is_active': source.is_active,
                    'last_sync': source.last_sync.isoformat() if source.last_sync else None,
                    'created_at': source.created_at.isoformat()
                })
            
            return configured_sources
            
        except Exception as e:
            print(f"Failed to get configured sources: {str(e)}")
            return []
    
    def sync_external_data(self, source_id: int, target_date_range: Dict = None) -> Dict:
        """Sync data from external source"""
        try:
            source = ExternalDataSource.query.get(source_id)
            if not source:
                raise ValueError("External data source not found")
            
            if not source.is_active:
                raise ValueError("Data source is inactive")
            
            # Get date range
            if target_date_range:
                start_date = target_date_range['start_date']
                end_date = target_date_range['end_date']
            else:
                # Default to last 30 days
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Fetch data based on source type
            configuration = source.get_api_configuration()
            
            if source.source_type == 'holidays':
                data = self.fetch_holiday_data(
                    configuration.get('country_code', 'US'),
                    start_date,
                    end_date
                )
            elif source.source_type == 'weather':
                data = self.fetch_weather_data(
                    configuration.get('location', 'New York'),
                    start_date,
                    end_date,
                    configuration.get('api_key')
                )
            elif source.source_type == 'economic':
                data = self.fetch_economic_data(
                    configuration.get('indicator', 'gdp'),
                    start_date,
                    end_date,
                    configuration.get('country', 'US')
                )
            elif source.source_type == 'events':
                data = self.fetch_event_data(
                    configuration.get('location', 'New York'),
                    start_date,
                    end_date,
                    configuration.get('event_type')
                )
            else:
                raise ValueError(f"Unknown source type: {source.source_type}")
            
            # Update last sync time
            source.last_sync = datetime.utcnow()
            db.session.commit()
            
            return {
                'success': True,
                'source_id': source_id,
                'source_type': source.source_type,
                'records_fetched': len(data),
                'date_range': {'start': start_date, 'end': end_date},
                'data_preview': data.head().to_dict('records') if not data.empty else [],
                'last_sync': source.last_sync.isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'source_id': source_id
            }
    
    def merge_external_data(self, main_df: pd.DataFrame, external_data: pd.DataFrame, 
                          merge_key: str = 'date', merge_type: str = 'left') -> pd.DataFrame:
        """Merge external data with main dataset"""
        try:
            if external_data.empty:
                return main_df
            
            # Ensure merge key is datetime
            if merge_key in main_df.columns:
                main_df[merge_key] = pd.to_datetime(main_df[merge_key])
            if merge_key in external_data.columns:
                external_data[merge_key] = pd.to_datetime(external_data[merge_key])
            
            # Perform merge
            merged_df = pd.merge(main_df, external_data, on=merge_key, how=merge_type)
            
            return merged_df
            
        except Exception as e:
            print(f"Failed to merge external data: {str(e)}")
            return main_df
    
    def get_data_summary(self) -> Dict:
        """Get summary of available and configured external data sources"""
        available_sources = self.get_available_sources()
        configured_sources = self.get_configured_sources()
        
        summary = {
            'available_sources': available_sources,
            'configured_sources_count': len(configured_sources),
            'active_sources_count': len([s for s in configured_sources if s['is_active']]),
            'source_types': list(available_sources.keys()),
            'last_sync_times': {}
        }
        
        # Add last sync times
        for source in configured_sources:
            if source['last_sync']:
                summary['last_sync_times'][source['source_type']] = source['last_sync']
        
        return summary
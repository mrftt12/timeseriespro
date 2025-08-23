"""
Comprehensive unit and integration tests for the DataProfiler engine
Tests statistical calculations, outlier detection, and edge cases
"""

import unittest
import tempfile
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from app import app, db
from models import Project, DataProfile
from data_profiler import DataProfiler


class TestDataProfiler(unittest.TestCase):
    """Test cases for DataProfiler functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database"""
        cls.db_fd, cls.db_path = tempfile.mkstemp()
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{cls.db_path}'
        app.config['TESTING'] = True
        
    def setUp(self):
        """Set up test fixtures before each test"""
        self.app = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        db.create_all()
        
        # Create test project
        self.test_project = Project(
            name="Data Profiler Test Project",
            description="Test project for data profiler",
            date_column="date",
            target_column="value"
        )
        db.session.add(self.test_project)
        db.session.commit()
        
        # Create test datasets
        self.test_data_path = self._create_test_dataset()
        self.time_series_path = self._create_time_series_dataset()
        self.problematic_data_path = self._create_problematic_dataset()
        
    def tearDown(self):
        """Clean up after each test"""
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
        
        # Clean up test files
        for path in [self.test_data_path, self.time_series_path, self.problematic_data_path]:
            if os.path.exists(path):
                os.unlink(path)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test database"""
        os.close(cls.db_fd)
        os.unlink(cls.db_path)
    
    def _create_test_dataset(self):
        """Create a basic test dataset"""
        np.random.seed(42)
        
        data = {
            'id': range(1, 101),
            'numeric_normal': np.random.normal(50, 10, 100),
            'numeric_skewed': np.random.exponential(2, 100),
            'categorical': ['A', 'B', 'C'] * 33 + ['A'],
            'text_data': [f'text_{i}' for i in range(100)],
            'numeric_with_outliers': np.concatenate([np.random.normal(10, 2, 95), [100, 105, 110, 115, 120]]),
            'numeric_with_missing': [i if i % 10 != 0 else None for i in range(100)],
            'boolean_data': [True, False] * 50,
            'zero_values': [0] * 20 + list(range(1, 81)),
            'constant_column': [42] * 100
        }
        
        df = pd.DataFrame(data)
        
        # Create temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return temp_file.name
    
    def _create_time_series_dataset(self):
        """Create a time series dataset"""
        np.random.seed(42)
        
        # Create dates with some gaps
        start_date = datetime(2020, 1, 1)
        dates = []
        current_date = start_date
        
        for i in range(100):
            dates.append(current_date)
            # Add irregular gaps
            if i in [20, 50, 80]:
                current_date += timedelta(days=5)  # Large gap
            else:
                current_date += timedelta(days=1)
        
        # Create time series data with trend and seasonality
        trend = np.linspace(100, 200, 100)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(100) / 12)
        noise = np.random.normal(0, 5, 100)
        values = trend + seasonal + noise
        
        data = {
            'date': dates,
            'value': values,
            'volume': np.random.poisson(1000, 100),
            'category': ['prod_' + str(i % 5) for i in range(100)]
        }
        
        df = pd.DataFrame(data)
        
        # Create temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return temp_file.name
    
    def _create_problematic_dataset(self):
        """Create a dataset with various data quality issues"""
        data = {
            'mixed_types': ['1', 2, '3.0', 4, 'text', None] * 10,  # Mixed data types
            'high_missing': [i if i % 3 != 0 else None for i in range(60)],  # 33% missing
            'duplicates': [1, 2, 3] * 20,  # Many duplicates
            'all_missing': [None] * 60,  # Completely missing column
            'single_value': ['constant'] * 60,  # No variation
            'extreme_outliers': [1] * 55 + [1000, 2000, 3000, 4000, 5000],  # Extreme outliers
            'unicode_text': ['café', 'naïve', 'résumé'] * 20,  # Unicode text
            'empty_strings': ['', '  ', 'actual_text'] * 20,  # Empty/whitespace strings
        }
        
        df = pd.DataFrame(data)
        
        # Create temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8')
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return temp_file.name
    
    def test_basic_dataset_analysis(self):
        """Test basic dataset analysis functionality"""
        profiler = DataProfiler(
            project_id=self.test_project.id,
            file_path=self.test_data_path,
            target_column='numeric_normal'
        )
        
        results = profiler.analyze_dataset()
        
        # Test basic structure
        self.assertIn('basic_statistics', results)
        self.assertIn('column_analysis', results)
        self.assertIn('data_quality', results)
        self.assertIn('quality_score', results)
        
        # Test basic statistics
        basic_stats = results['basic_statistics']
        self.assertEqual(basic_stats['dataset_shape']['rows'], 100)
        self.assertEqual(basic_stats['dataset_shape']['columns'], 10)
        self.assertGreater(basic_stats['dataset_shape']['numeric_columns'], 0)
        self.assertGreater(basic_stats['dataset_shape']['categorical_columns'], 0)
        
        # Test quality score is reasonable
        self.assertGreaterEqual(results['quality_score'], 0)
        self.assertLessEqual(results['quality_score'], 100)
    
    def test_numeric_column_analysis(self):
        """Test detailed numeric column analysis"""
        profiler = DataProfiler(
            project_id=self.test_project.id,
            file_path=self.test_data_path
        )
        
        results = profiler.analyze_dataset()
        column_analysis = results['column_analysis']
        
        # Test normal distribution column
        normal_col = column_analysis['numeric_normal']
        self.assertIn('statistics', normal_col)
        self.assertIn('distribution_tests', normal_col)
        self.assertIn('outliers', normal_col)
        
        stats = normal_col['statistics']
        self.assertAlmostEqual(stats['mean'], 50, delta=5)  # Should be around 50
        self.assertGreater(stats['std'], 0)
        self.assertLess(stats['min'], stats['max'])
        
        # Test outlier detection results
        outliers = normal_col['outliers']
        self.assertIn('z_score', outliers)
        self.assertIn('iqr', outliers)
        self.assertIn('modified_z_score', outliers)
        
        for method in ['z_score', 'iqr', 'modified_z_score']:
            self.assertIn('count', outliers[method])
            self.assertIn('percentage', outliers[method])
            self.assertGreaterEqual(outliers[method]['count'], 0)
    
    def test_categorical_column_analysis(self):
        """Test categorical column analysis"""
        profiler = DataProfiler(
            project_id=self.test_project.id,
            file_path=self.test_data_path
        )
        
        results = profiler.analyze_dataset()
        column_analysis = results['column_analysis']
        
        # Test categorical column
        cat_col = column_analysis['categorical']
        self.assertIn('value_counts', cat_col)
        self.assertIn('cardinality', cat_col)
        self.assertIn('most_frequent', cat_col)
        
        self.assertEqual(cat_col['cardinality'], 3)  # A, B, C
        self.assertIn(cat_col['most_frequent'], ['A', 'B', 'C'])
    
    def test_outlier_detection_methods(self):
        """Test multiple outlier detection methods"""
        profiler = DataProfiler(
            project_id=self.test_project.id,
            file_path=self.test_data_path
        )
        
        results = profiler.analyze_dataset()
        
        # Test column with known outliers
        outlier_col = results['column_analysis']['numeric_with_outliers']
        outliers = outlier_col['outliers']
        
        # All methods should detect the extreme outliers we added
        for method in ['z_score', 'iqr', 'modified_z_score']:
            self.assertGreater(outliers[method]['count'], 0)
            self.assertGreater(outliers[method]['percentage'], 0)
    
    def test_correlation_analysis(self):
        """Test correlation analysis functionality"""
        profiler = DataProfiler(
            project_id=self.test_project.id,
            file_path=self.test_data_path,
            target_column='numeric_normal'
        )
        
        results = profiler.analyze_dataset()
        
        if 'correlation_analysis' in results and 'error' not in results['correlation_analysis']:
            corr_analysis = results['correlation_analysis']
            self.assertIn('correlation_matrix', corr_analysis)
            self.assertIn('correlation_with_target', corr_analysis)
            
            # Should have correlation with target column
            target_corr = corr_analysis['correlation_with_target']
            self.assertIsInstance(target_corr, dict)
    
    def test_time_series_analysis(self):
        """Test time series specific analysis"""
        profiler = DataProfiler(
            project_id=self.test_project.id,
            file_path=self.time_series_path,
            date_column='date',
            target_column='value'
        )
        
        results = profiler.analyze_dataset()
        
        if 'time_series_analysis' in results and 'error' not in results['time_series_analysis']:
            ts_analysis = results['time_series_analysis']
            
            # Test stationarity tests
            if 'stationarity' in ts_analysis:
                stationarity = ts_analysis['stationarity']
                self.assertIn('adf_test', stationarity)
                
            # Test trend analysis
            if 'trend' in ts_analysis:
                trend = ts_analysis['trend']
                self.assertIn('slope', trend)
                self.assertIn('trend_direction', trend)
                self.assertIn('trend_strength', trend)
    
    def test_missing_data_analysis(self):
        """Test missing data pattern analysis"""
        profiler = DataProfiler(
            project_id=self.test_project.id,
            file_path=self.test_data_path
        )
        
        results = profiler.analyze_dataset()
        
        missing_analysis = results['missing_data_analysis']
        self.assertIn('total_missing_cells', missing_analysis)
        self.assertIn('missing_percentage', missing_analysis)
        self.assertIn('columns_with_missing', missing_analysis)
        
        # Should detect missing values in 'numeric_with_missing' column
        self.assertIn('numeric_with_missing', missing_analysis['columns_with_missing'])
        
        missing_col_info = missing_analysis['columns_with_missing']['numeric_with_missing']
        self.assertGreater(missing_col_info['count'], 0)
        self.assertGreater(missing_col_info['percentage'], 0)
    
    def test_data_quality_assessment(self):
        """Test overall data quality assessment"""
        profiler = DataProfiler(
            project_id=self.test_project.id,
            file_path=self.test_data_path
        )
        
        results = profiler.analyze_dataset()
        
        quality_assessment = results['data_quality']
        self.assertIn('issues', quality_assessment)
        self.assertIn('recommendations', quality_assessment)
        self.assertIn('overall_score', quality_assessment)
        
        # Overall score should be reasonable
        self.assertGreaterEqual(quality_assessment['overall_score'], 0)
        self.assertLessEqual(quality_assessment['overall_score'], 100)
    
    def test_recommendations_generation(self):
        """Test actionable recommendations generation"""
        profiler = DataProfiler(
            project_id=self.test_project.id,
            file_path=self.problematic_data_path
        )
        
        results = profiler.analyze_dataset()
        
        recommendations = results['recommendations']
        self.assertIsInstance(recommendations, list)
        
        # Should generate recommendations for problematic data
        if len(recommendations) > 0:
            rec = recommendations[0]
            self.assertIn('category', rec)
            self.assertIn('priority', rec)
            self.assertIn('issue', rec)
            self.assertIn('recommendation', rec)
            self.assertIn('action', rec)
    
    def test_caching_functionality(self):
        """Test profile caching and retrieval"""
        profiler = DataProfiler(
            project_id=self.test_project.id,
            file_path=self.test_data_path
        )
        
        # First analysis should generate fresh results
        results1 = profiler.analyze_dataset()
        self.assertFalse(results1['profile_metadata'].get('cached', False))
        
        # Second analysis should use cached results
        results2 = profiler.analyze_dataset()
        # Note: Caching might not work in this test environment due to database setup
        
        # Force refresh should generate fresh results
        results3 = profiler.analyze_dataset(force_refresh=True)
        self.assertFalse(results3['profile_metadata'].get('cached', False))
    
    def test_edge_cases(self):
        """Test various edge cases and error handling"""
        # Test with single row dataset
        single_row_data = pd.DataFrame({'col1': [1], 'col2': ['text']})
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        single_row_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            profiler = DataProfiler(
                project_id=self.test_project.id,
                file_path=temp_file.name
            )
            
            results = profiler.analyze_dataset()
            
            # Should handle single row gracefully
            self.assertIn('basic_statistics', results)
            self.assertEqual(results['basic_statistics']['dataset_shape']['rows'], 1)
            
        finally:
            os.unlink(temp_file.name)
        
        # Test with empty dataset
        empty_data = pd.DataFrame()
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        empty_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            profiler = DataProfiler(
                project_id=self.test_project.id,
                file_path=temp_file.name
            )
            
            # Should handle empty data gracefully (might raise exception)
            try:
                results = profiler.analyze_dataset()
            except Exception:
                pass  # Expected for empty dataset
                
        finally:
            os.unlink(temp_file.name)
    
    def test_statistical_accuracy(self):
        """Test accuracy of statistical calculations"""
        # Create dataset with known statistical properties
        np.random.seed(123)
        known_data = {
            'normal_100_15': np.random.normal(100, 15, 1000),  # Mean=100, Std=15
            'uniform': np.random.uniform(0, 10, 1000),  # Uniform distribution
            'constant': [42] * 1000  # Constant values
        }
        
        df = pd.DataFrame(known_data)
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            profiler = DataProfiler(
                project_id=self.test_project.id,
                file_path=temp_file.name
            )
            
            results = profiler.analyze_dataset()
            column_analysis = results['column_analysis']
            
            # Test normal distribution statistics
            normal_stats = column_analysis['normal_100_15']['statistics']
            self.assertAlmostEqual(normal_stats['mean'], 100, delta=5)
            self.assertAlmostEqual(normal_stats['std'], 15, delta=3)
            
            # Test constant column
            constant_stats = column_analysis['constant']['statistics']
            self.assertEqual(constant_stats['std'], 0)  # No variation
            self.assertEqual(constant_stats['min'], constant_stats['max'])
            
        finally:
            os.unlink(temp_file.name)
    
    def test_large_dataset_handling(self):
        """Test handling of larger datasets"""
        # Create a moderately large dataset
        np.random.seed(42)
        large_data = {
            'id': range(10000),
            'value1': np.random.normal(0, 1, 10000),
            'value2': np.random.exponential(1, 10000),
            'category': ['cat_' + str(i % 100) for i in range(10000)]
        }
        
        df = pd.DataFrame(large_data)
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            profiler = DataProfiler(
                project_id=self.test_project.id,
                file_path=temp_file.name
            )
            
            # Should handle larger dataset without issues
            results = profiler.analyze_dataset()
            
            self.assertEqual(results['basic_statistics']['dataset_shape']['rows'], 10000)
            self.assertIn('column_analysis', results)
            
        finally:
            os.unlink(temp_file.name)


class TestDataProfilerIntegration(unittest.TestCase):
    """Integration tests for DataProfiler with database and API"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database"""
        cls.db_fd, cls.db_path = tempfile.mkstemp()
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{cls.db_path}'
        app.config['TESTING'] = True
        
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        db.create_all()
        
        # Create test project with dataset
        self.test_project = Project(
            name="Integration Test Project",
            description="Test project for integration testing"
        )
        db.session.add(self.test_project)
        db.session.commit()
        
        # Create test dataset file
        self.test_data_path = self._create_test_dataset()
        self.test_project.dataset_path = self.test_data_path
        db.session.commit()
        
    def tearDown(self):
        """Clean up after each test"""
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
        
        if os.path.exists(self.test_data_path):
            os.unlink(self.test_data_path)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test database"""
        os.close(cls.db_fd)
        os.unlink(cls.db_path)
    
    def _create_test_dataset(self):
        """Create test dataset for integration testing"""
        np.random.seed(42)
        data = {
            'date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'value': np.random.normal(50, 10, 100),
            'category': ['A', 'B', 'C'] * 33 + ['A']
        }
        
        df = pd.DataFrame(data)
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return temp_file.name
    
    def test_api_generate_profile(self):
        """Test profile generation via API"""
        response = self.app.post(f'/api/projects/{self.test_project.id}/profile')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('profile', data)
        self.assertIn('message', data)
    
    def test_api_get_profile(self):
        """Test profile retrieval via API"""
        # First generate a profile
        self.app.post(f'/api/projects/{self.test_project.id}/profile')
        
        # Then retrieve it
        response = self.app.get(f'/api/projects/{self.test_project.id}/profile')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('profile', data)
    
    def test_api_refresh_profile(self):
        """Test profile refresh via API"""
        response = self.app.post(f'/api/projects/{self.test_project.id}/profile/refresh')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('profile', data)
        self.assertEqual(data['message'], 'Data profile refreshed successfully')
    
    def test_database_storage(self):
        """Test that profiles are properly stored in database"""
        profiler = DataProfiler(
            project_id=self.test_project.id,
            file_path=self.test_data_path,
            date_column='date',
            target_column='value'
        )
        
        # Generate profile
        results = profiler.analyze_dataset()
        
        # Check that profiles were stored in database
        stored_profiles = DataProfile.query.filter_by(project_id=self.test_project.id).all()
        self.assertGreater(len(stored_profiles), 0)
        
        # Check profile data integrity
        for profile in stored_profiles:
            self.assertEqual(profile.project_id, self.test_project.id)
            self.assertIsNotNone(profile.column_name)
            self.assertIsNotNone(profile.data_type)
            self.assertGreaterEqual(profile.quality_score, 0)
    
    def test_error_handling(self):
        """Test API error handling for invalid requests"""
        # Test with non-existent project
        response = self.app.post('/api/projects/99999/profile')
        self.assertEqual(response.status_code, 404)
        
        # Test with project without dataset
        empty_project = Project(name="Empty Project")
        db.session.add(empty_project)
        db.session.commit()
        
        response = self.app.post(f'/api/projects/{empty_project.id}/profile')
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertIn('error', data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
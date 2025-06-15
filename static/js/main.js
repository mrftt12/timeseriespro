// TimeSeries Forecasting Platform JavaScript

// Global utilities
const ForecastingApp = {
    // Initialize the application
    init: function() {
        this.setupEventListeners();
        this.initializeTooltips();
    },

    // Setup global event listeners
    setupEventListeners: function() {
        // Auto-dismiss alerts after 5 seconds
        document.querySelectorAll('.alert').forEach(alert => {
            if (!alert.classList.contains('alert-danger')) {
                setTimeout(() => {
                    const bsAlert = new bootstrap.Alert(alert);
                    bsAlert.close();
                }, 5000);
            }
        });

        // Handle form loading states
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', function(e) {
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn && !submitBtn.disabled) {
                    this.setLoadingState(submitBtn);
                }
            }.bind(this));
        });
    },

    // Initialize Bootstrap tooltips
    initializeTooltips: function() {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    },

    // Set loading state for buttons
    setLoadingState: function(button) {
        const originalText = button.innerHTML;
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        
        // Store original text for potential restoration
        button.dataset.originalText = originalText;
    },

    // Restore button from loading state
    restoreButtonState: function(button) {
        if (button.dataset.originalText) {
            button.innerHTML = button.dataset.originalText;
            button.disabled = false;
        }
    },

    // Format numbers for display
    formatNumber: function(num, decimals = 4) {
        if (num === null || num === undefined) return 'N/A';
        return parseFloat(num).toFixed(decimals);
    },

    // Format file size
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    // Show notification
    showNotification: function(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.style.position = 'fixed';
        alertDiv.style.top = '20px';
        alertDiv.style.right = '20px';
        alertDiv.style.zIndex = '9999';
        alertDiv.style.minWidth = '300px';
        
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 5000);
    },

    // Validate file type
    validateFileType: function(file, allowedTypes = ['csv', 'xlsx', 'xls']) {
        const fileExtension = file.name.split('.').pop().toLowerCase();
        return allowedTypes.includes(fileExtension);
    },

    // Handle AJAX errors
    handleAjaxError: function(error) {
        console.error('AJAX Error:', error);
        this.showNotification('An error occurred while processing your request.', 'danger');
    }
};

// Table utilities
const TableUtils = {
    // Make table sortable
    makeSortable: function(tableId) {
        const table = document.getElementById(tableId);
        if (!table) return;

        const headers = table.querySelectorAll('th[data-sortable]');
        headers.forEach(header => {
            header.style.cursor = 'pointer';
            header.innerHTML += ' <i class="fas fa-sort text-muted"></i>';
            
            header.addEventListener('click', () => {
                this.sortTable(table, header.cellIndex, header.dataset.type || 'string');
            });
        });
    },

    // Sort table by column
    sortTable: function(table, columnIndex, dataType = 'string') {
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        
        const sortedRows = rows.sort((a, b) => {
            const aValue = a.cells[columnIndex].textContent.trim();
            const bValue = b.cells[columnIndex].textContent.trim();
            
            if (dataType === 'number') {
                return parseFloat(aValue) - parseFloat(bValue);
            } else if (dataType === 'date') {
                return new Date(aValue) - new Date(bValue);
            } else {
                return aValue.localeCompare(bValue);
            }
        });
        
        // Clear tbody and append sorted rows
        tbody.innerHTML = '';
        sortedRows.forEach(row => tbody.appendChild(row));
    },

    // Filter table rows
    filterTable: function(tableId, searchTerm) {
        const table = document.getElementById(tableId);
        if (!table) return;

        const rows = table.querySelectorAll('tbody tr');
        const searchLower = searchTerm.toLowerCase();

        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            row.style.display = text.includes(searchLower) ? '' : 'none';
        });
    }
};

// Form utilities
const FormUtils = {
    // Validate form fields
    validateForm: function(formId) {
        const form = document.getElementById(formId);
        if (!form) return false;

        let isValid = true;
        const requiredFields = form.querySelectorAll('[required]');

        requiredFields.forEach(field => {
            if (!field.value.trim()) {
                this.showFieldError(field, 'This field is required');
                isValid = false;
            } else {
                this.clearFieldError(field);
            }
        });

        return isValid;
    },

    // Show field error
    showFieldError: function(field, message) {
        this.clearFieldError(field);
        
        field.classList.add('is-invalid');
        const errorDiv = document.createElement('div');
        errorDiv.className = 'invalid-feedback';
        errorDiv.textContent = message;
        field.parentNode.appendChild(errorDiv);
    },

    // Clear field error
    clearFieldError: function(field) {
        field.classList.remove('is-invalid');
        const errorDiv = field.parentNode.querySelector('.invalid-feedback');
        if (errorDiv) {
            errorDiv.remove();
        }
    },

    // Auto-save form data
    autoSave: function(formId, storageKey) {
        const form = document.getElementById(formId);
        if (!form) return;

        // Load saved data
        const savedData = localStorage.getItem(storageKey);
        if (savedData) {
            const data = JSON.parse(savedData);
            Object.keys(data).forEach(key => {
                const field = form.querySelector(`[name="${key}"]`);
                if (field) field.value = data[key];
            });
        }

        // Save data on input
        form.addEventListener('input', () => {
            const formData = new FormData(form);
            const data = Object.fromEntries(formData.entries());
            localStorage.setItem(storageKey, JSON.stringify(data));
        });

        // Clear saved data on successful submit
        form.addEventListener('submit', () => {
            localStorage.removeItem(storageKey);
        });
    }
};

// Data processing utilities
const DataUtils = {
    // Parse CSV text
    parseCSV: function(csvText) {
        const lines = csvText.split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        const data = [];

        for (let i = 1; i < lines.length; i++) {
            if (lines[i].trim()) {
                const values = lines[i].split(',').map(v => v.trim());
                const row = {};
                headers.forEach((header, index) => {
                    row[header] = values[index] || '';
                });
                data.push(row);
            }
        }

        return { headers, data };
    },

    // Detect data types
    detectDataType: function(values) {
        const nonEmptyValues = values.filter(v => v !== null && v !== '' && v !== undefined);
        if (nonEmptyValues.length === 0) return 'string';

        // Check if all values are numeric
        const numericValues = nonEmptyValues.filter(v => !isNaN(parseFloat(v)));
        if (numericValues.length === nonEmptyValues.length) {
            return 'number';
        }

        // Check if values look like dates
        const dateValues = nonEmptyValues.filter(v => !isNaN(Date.parse(v)));
        if (dateValues.length === nonEmptyValues.length) {
            return 'date';
        }

        return 'string';
    },

    // Generate data summary
    generateSummary: function(data, column) {
        const values = data.map(row => row[column]).filter(v => v !== '' && v !== null);
        
        if (values.length === 0) {
            return { count: 0, type: 'empty' };
        }

        const dataType = this.detectDataType(values);
        const summary = {
            count: values.length,
            type: dataType,
            missing: data.length - values.length
        };

        if (dataType === 'number') {
            const numValues = values.map(v => parseFloat(v));
            summary.min = Math.min(...numValues);
            summary.max = Math.max(...numValues);
            summary.mean = numValues.reduce((a, b) => a + b, 0) / numValues.length;
            summary.std = Math.sqrt(numValues.reduce((sq, n) => sq + Math.pow(n - summary.mean, 2), 0) / numValues.length);
        }

        return summary;
    }
};

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    ForecastingApp.init();
    
    // Initialize any sortable tables
    document.querySelectorAll('table[data-sortable]').forEach(table => {
        TableUtils.makeSortable(table.id);
    });
});

// Export utilities for use in other scripts
window.ForecastingApp = ForecastingApp;
window.TableUtils = TableUtils;
window.FormUtils = FormUtils;
window.DataUtils = DataUtils;

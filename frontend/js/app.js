/**
 * CLV Prediction Dashboard - JavaScript Application
 * Handles API communication, chart rendering, and UI interactions
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000';

// State
let currentPage = 1;
const pageSize = 20;
let currentSegmentFilter = '';
let dashboardData = null;
let charts = {};

// DOM Ready
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

// Initialize Application
function initializeApp() {
    // Navigation
    setupNavigation();

    // Event listeners
    setupEventListeners();

    // Check API and load initial data
    checkApiHealth();
    loadDashboardData();
}

// Navigation Setup
function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();

            // Update active state
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');

            // Show corresponding section
            const section = item.dataset.section;
            showSection(section);

            // Update page title
            const titles = {
                'dashboard': 'Dashboard',
                'customers': 'Customer Directory',
                'segments': 'Customer Segments',
                'meta-ads': 'Meta Ads Optimization',
                'predictions': 'Predict CLV'
            };
            document.getElementById('pageTitle').textContent = titles[section] || 'Dashboard';
        });
    });
}

// Show Section
function showSection(sectionName) {
    const sections = document.querySelectorAll('.content-section');
    sections.forEach(section => section.classList.remove('active'));

    const targetSection = document.getElementById(`${sectionName}-section`);
    if (targetSection) {
        targetSection.classList.add('active');

        // Show/hide search box based on section
        const searchBox = document.getElementById('searchBox');
        if (sectionName === 'customers') {
            searchBox.style.display = 'flex';
            document.getElementById('searchInput').value = '';
        } else {
            searchBox.style.display = 'none';
        }

        // Load section-specific data
        switch (sectionName) {
            case 'customers':
                loadCustomers();
                break;
            case 'segments':
                loadSegments();
                break;
            case 'meta-ads':
                loadMetaAdsData();
                break;
        }
    }
}

// Event Listeners
function setupEventListeners() {
    // Refresh button
    document.getElementById('refreshBtn').addEventListener('click', async () => {
        const btn = document.getElementById('refreshBtn');
        btn.disabled = true;
        btn.textContent = '⏳';
        showNotification('Refreshing data...', 'info');

        try {
            // Clear cache first
            try {
                await fetch(`${API_BASE_URL}/api/cache/clear`, { method: 'POST' });
            } catch (e) {
                console.log('Cache clear not available');
            }

            // Reload dashboard data
            await loadDashboardData();
            showNotification('Dashboard refreshed successfully!', 'success');
        } catch (error) {
            showNotification('Failed to refresh data', 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = '🔄';
        }
    });

    // Segment filter
    // Segment filter - works with search
    document.getElementById('segmentFilter').addEventListener('change', (e) => {
        currentSegmentFilter = e.target.value;
        currentPage = 1;
        const searchTerm = document.getElementById('searchInput').value.trim();
        if (searchTerm.length > 0) {
            // Both search and segment filter
            searchCustomerWithFilter(searchTerm, currentSegmentFilter);
        } else {
            // Just segment filter
            loadCustomers();
        }
    });

    // Pagination
    document.getElementById('prevPage').addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            loadCustomers();
        }
    });

    document.getElementById('nextPage').addEventListener('click', () => {
        currentPage++;
        loadCustomers();
    });

    // Budget calculation
    document.getElementById('calculateBudget').addEventListener('click', () => {
        const budget = parseFloat(document.getElementById('budgetInput').value);
        loadBudgetAllocation(budget);
    });

    // Prediction form
    document.getElementById('predictionForm').addEventListener('submit', (e) => {
        e.preventDefault();
        submitPrediction();
    });

    // Search - works with segment filter
    document.getElementById('searchInput').addEventListener('input', debounce((e) => {
        const searchTerm = e.target.value.trim();
        if (searchTerm.length > 0) {
            searchCustomerWithFilter(searchTerm, currentSegmentFilter);
        } else {
            // No search term - show all customers (with segment filter if selected)
            loadCustomers();
        }
    }, 300));

    // Also handle Enter key for search
    document.getElementById('searchInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const searchTerm = e.target.value.trim();
            if (searchTerm.length > 0) {
                searchCustomer(searchTerm);
            }
        }
    });
}

// API Health Check
async function checkApiHealth() {
    const statusDot = document.getElementById('apiStatusDot');
    const statusText = document.getElementById('apiStatusText');

    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            statusDot.classList.add('online');
            statusDot.classList.remove('offline');
            statusText.textContent = 'API Online';
        }
    } catch (error) {
        statusDot.classList.add('offline');
        statusDot.classList.remove('online');
        statusText.textContent = 'API Offline';
        console.error('API health check failed:', error);
    }
}

// Load Dashboard Data
async function loadDashboardData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/dashboard/summary`);

        if (!response.ok) {
            // Use demo data if API is not available
            dashboardData = getDemoData();
        } else {
            dashboardData = await response.json();
        }

        updateDashboardUI(dashboardData);
        renderCharts(dashboardData);

    } catch (error) {
        console.error('Failed to load dashboard data:', error);
        // Use demo data
        dashboardData = getDemoData();
        updateDashboardUI(dashboardData);
        renderCharts(dashboardData);
    }
}

// Update Dashboard UI
function updateDashboardUI(data) {
    // KPI Cards
    document.getElementById('totalCustomers').textContent = formatNumber(data.summary.total_customers);
    document.getElementById('avgClv').textContent = formatCurrency(data.summary.avg_clv);
    document.getElementById('highClvRate').textContent = `${data.summary.high_clv_percentage}%`;
    document.getElementById('totalValue').textContent = formatCurrency(data.summary.total_value);

    // Top Customers Table
    const tbody = document.querySelector('#topCustomersTable tbody');
    tbody.innerHTML = '';

    data.top_customers.forEach(customer => {
        const clvKey = customer.predicted_clv || customer.actual_clv;
        const segmentKey = customer.predicted_segment || customer.customer_segment;
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${customer.customer_id}</td>
            <td>${formatCurrency(clvKey)}</td>
            <td><span class="segment-badge ${getSegmentClass(segmentKey)}">${segmentKey}</span></td>
            <td>${customer.total_orders}</td>
            <td>${customer.acquisition_source}</td>
        `;
        tbody.appendChild(row);
    });
}

// Render Charts
function renderCharts(data) {
    // Destroy existing charts
    Object.values(charts).forEach(chart => chart.destroy());
    charts = {};

    // CLV Distribution Chart
    const clvCtx = document.getElementById('clvDistributionChart').getContext('2d');
    charts.clvDistribution = new Chart(clvCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(data.clv_distribution),
            datasets: [{
                label: 'Customers',
                data: Object.values(data.clv_distribution),
                backgroundColor: [
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(99, 102, 241, 0.8)',
                    'rgba(139, 92, 246, 0.8)',
                    'rgba(16, 185, 129, 0.8)'
                ],
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#9ca3af'
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#9ca3af'
                    }
                }
            }
        }
    });

    // Segment Chart
    const segmentCtx = document.getElementById('segmentChart').getContext('2d');
    charts.segment = new Chart(segmentCtx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(data.segments),
            datasets: [{
                data: Object.values(data.segments),
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(99, 102, 241, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#9ca3af',
                        padding: 20
                    }
                }
            }
        }
    });

    // Acquisition Chart
    const acqCtx = document.getElementById('acquisitionChart').getContext('2d');
    charts.acquisition = new Chart(acqCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(data.acquisition_sources),
            datasets: [{
                label: 'Customers',
                data: Object.values(data.acquisition_sources),
                backgroundColor: 'rgba(99, 102, 241, 0.8)',
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#9ca3af'
                    }
                },
                y: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#9ca3af'
                    }
                }
            }
        }
    });
}

// Load Customers
async function loadCustomers() {
    const tbody = document.querySelector('#customersTable tbody');
    tbody.innerHTML = '<tr><td colspan="7" class="loading">Loading...</td></tr>';

    try {
        let url = `${API_BASE_URL}/api/customers?limit=${pageSize}&offset=${(currentPage - 1) * pageSize}`;
        if (currentSegmentFilter) {
            url += `&segment=${encodeURIComponent(currentSegmentFilter)}`;
        }

        const response = await fetch(url);

        if (!response.ok) {
            throw new Error('API not available');
        }

        const data = await response.json();
        renderCustomersTable(data.customers);
        updatePagination(data.total);

    } catch (error) {
        console.error('Failed to load customers:', error);
        // Use demo data
        const demoCustomers = getDemoCustomers();
        renderCustomersTable(demoCustomers);
        updatePagination(demoCustomers.length);
    }
}

// Render Customers Table
function renderCustomersTable(customers) {
    const tbody = document.querySelector('#customersTable tbody');
    tbody.innerHTML = '';

    customers.forEach(customer => {
        const clv = customer.predicted_clv || customer.actual_clv || 0;
        const segment = customer.predicted_segment || customer.customer_segment || '-';

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${customer.customer_id}</td>
            <td>${formatCurrency(customer.total_spent)}</td>
            <td>${formatCurrency(clv)}</td>
            <td>${customer.total_orders}</td>
            <td><span class="segment-badge ${getSegmentClass(segment)}">${segment}</span></td>
            <td>${customer.acquisition_source}</td>
            <td>${(customer.email_engagement_rate * 100).toFixed(0)}%</td>
        `;
        tbody.appendChild(row);
    });
}

// Search Customer by ID with Segment Filter
async function searchCustomerWithFilter(searchTerm, segmentFilter = '') {
    const tbody = document.querySelector('#customersTable tbody');
    tbody.innerHTML = '<tr><td colspan="7" class="loading">Searching...</td></tr>';

    try {
        // Try to get specific customer by exact ID first
        const response = await fetch(`${API_BASE_URL}/api/customers/${encodeURIComponent(searchTerm)}`);

        if (response.ok) {
            const customer = await response.json();
            const segment = customer.predicted_segment || customer.customer_segment || '';

            // Check if customer matches segment filter (if specified)
            if (!segmentFilter || segment === segmentFilter) {
                renderCustomersTable([customer]);
                updatePagination(1);
                showNotification(`Found customer: ${searchTerm}`, 'success');
            } else {
                tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 40px; color: #f59e0b; font-size: 1.1em;">
                    <strong>⚠️ Customer "${searchTerm}" Found But Not in ${segmentFilter} Segment</strong>
                    <br><span style="color: #9ca3af; font-size: 0.9em;">This customer is in the "${segment}" segment.</span>
                </td></tr>`;
                updatePagination(0);
                showNotification(`${searchTerm} is in ${segment}, not ${segmentFilter}`, 'warning');
            }
            return;
        }

        // If not found by exact ID, fetch all and filter client-side
        let url = `${API_BASE_URL}/api/customers?limit=1000`;
        if (segmentFilter) {
            url += `&segment=${encodeURIComponent(segmentFilter)}`;
        }

        const allResponse = await fetch(url);

        if (allResponse.ok) {
            const data = await allResponse.json();
            const filtered = data.customers.filter(c =>
                c.customer_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
                (c.acquisition_source && c.acquisition_source.toLowerCase().includes(searchTerm.toLowerCase()))
            );

            if (filtered.length > 0) {
                renderCustomersTable(filtered);
                updatePagination(filtered.length);
                const msg = segmentFilter
                    ? `Found ${filtered.length} matching customers in ${segmentFilter}`
                    : `Found ${filtered.length} matching customers`;
                showNotification(msg, 'success');
            } else {
                tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 40px; color: #f59e0b; font-size: 1.1em;">
                    <strong>⚠️ No Customer ID "${searchTerm}" Found in Customer Directory</strong>
                    <br><span style="color: #9ca3af; font-size: 0.9em;">Please check the Customer ID and try again.</span>
                </td></tr>`;
                updatePagination(0);
                showNotification(`No Customer ID "${searchTerm}" found in directory`, 'warning');
            }
        }

    } catch (error) {
        console.error('Search failed:', error);
        tbody.innerHTML = '<tr><td colspan="7" class="loading">Search failed. Please try again.</td></tr>';
        showNotification('Search failed. Please try again.', 'error');
    }
}

// Legacy function - redirects to new one
async function searchCustomer(searchTerm) {
    return searchCustomerWithFilter(searchTerm, currentSegmentFilter);
}

// Show Notification
function showNotification(message, type = 'info') {
    // Create notification element if it doesn't exist
    let notification = document.getElementById('notification');
    if (!notification) {
        notification = document.createElement('div');
        notification.id = 'notification';
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 10000;
            opacity: 0;
            transition: opacity 0.3s ease-in-out;
            max-width: 400px;
        `;
        document.body.appendChild(notification);
    }

    // Set color based on type
    const colors = {
        success: '#10b981',
        warning: '#f59e0b',
        error: '#ef4444',
        info: '#6366f1'
    };
    notification.style.backgroundColor = colors[type] || colors.info;
    notification.textContent = message;
    notification.style.opacity = '1';

    // Hide after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
    }, 3000);
}

// Update Pagination
function updatePagination(total) {
    const totalPages = Math.ceil(total / pageSize);
    document.getElementById('pageInfo').textContent = `Page ${currentPage} of ${totalPages}`;
    document.getElementById('prevPage').disabled = currentPage <= 1;
    document.getElementById('nextPage').disabled = currentPage >= totalPages;
}

// Load Segments
async function loadSegments() {
    try {
        // Load segment data
        const response = await fetch(`${API_BASE_URL}/api/segments`);

        let segmentData;
        if (!response.ok) {
            segmentData = getDemoSegments();
        } else {
            segmentData = await response.json();
        }

        updateSegmentsUI(segmentData);

        // Load metrics for feature importance
        const metricsResponse = await fetch(`${API_BASE_URL}/api/metrics`);
        if (metricsResponse.ok) {
            const metricsData = await metricsResponse.json();
            renderFeatureImportanceChart(metricsData.feature_importance);
        } else {
            renderFeatureImportanceChart(getDemoFeatureImportance());
        }

    } catch (error) {
        console.error('Failed to load segments:', error);
        updateSegmentsUI(getDemoSegments());
        renderFeatureImportanceChart(getDemoFeatureImportance());
    }
}

// Update Segments UI
function updateSegmentsUI(data) {
    const segments = data.segments;

    // High-CLV
    if (segments['High-CLV']) {
        document.getElementById('highClvCount').textContent = formatNumber(segments['High-CLV'].count);
        document.getElementById('highClvAvg').textContent = formatCurrency(segments['High-CLV'].avg_predicted_clv || segments['High-CLV'].avg_clv);
        document.getElementById('highClvTotal').textContent = formatCurrency(segments['High-CLV'].total_predicted_value || segments['High-CLV'].total_value);
    }

    // Growth-Potential
    if (segments['Growth-Potential']) {
        document.getElementById('growthCount').textContent = formatNumber(segments['Growth-Potential'].count);
        document.getElementById('growthAvg').textContent = formatCurrency(segments['Growth-Potential'].avg_predicted_clv || segments['Growth-Potential'].avg_clv);
        document.getElementById('growthTotal').textContent = formatCurrency(segments['Growth-Potential'].total_predicted_value || segments['Growth-Potential'].total_value);
    }

    // Low-CLV
    if (segments['Low-CLV']) {
        document.getElementById('lowClvCount').textContent = formatNumber(segments['Low-CLV'].count);
        document.getElementById('lowClvAvg').textContent = formatCurrency(segments['Low-CLV'].avg_predicted_clv || segments['Low-CLV'].avg_clv);
        document.getElementById('lowClvTotal').textContent = formatCurrency(segments['Low-CLV'].total_predicted_value || segments['Low-CLV'].total_value);
    }
}

// Render Feature Importance Chart
function renderFeatureImportanceChart(features) {
    if (charts.featureImportance) {
        charts.featureImportance.destroy();
    }

    const ctx = document.getElementById('featureImportanceChart').getContext('2d');

    const labels = features.map(f => f.feature);
    const values = features.map(f => f.importance);

    charts.featureImportance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance',
                data: values,
                backgroundColor: 'rgba(99, 102, 241, 0.8)',
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#9ca3af'
                    }
                },
                y: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#9ca3af'
                    }
                }
            }
        }
    });
}

// Load Meta Ads Data
async function loadMetaAdsData() {
    const budget = parseFloat(document.getElementById('budgetInput').value);
    await loadBudgetAllocation(budget);
}

// Load Budget Allocation
async function loadBudgetAllocation(budget) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/meta-ads/budget-allocation?total_budget=${budget}`);

        let data;
        if (!response.ok) {
            data = getDemoBudgetAllocation(budget);
        } else {
            data = await response.json();
        }

        updateBudgetUI(data);
        renderBudgetChart(data);

    } catch (error) {
        console.error('Failed to load budget allocation:', error);
        const data = getDemoBudgetAllocation(budget);
        updateBudgetUI(data);
        renderBudgetChart(data);
    }
}

// Update Budget UI
function updateBudgetUI(data) {
    const allocation = data.allocation;

    // High-CLV
    document.getElementById('highClvBudget').textContent = formatCurrency(allocation['High-CLV'].budget);
    document.getElementById('highClvAcq').textContent = formatNumber(allocation['High-CLV'].expected_acquisitions);
    document.getElementById('highClvRoas').textContent = `${allocation['High-CLV'].expected_roas}x`;

    // Growth-Potential
    document.getElementById('growthBudget').textContent = formatCurrency(allocation['Growth-Potential'].budget);
    document.getElementById('growthAcq').textContent = formatNumber(allocation['Growth-Potential'].expected_acquisitions);
    document.getElementById('growthRoas').textContent = `${allocation['Growth-Potential'].expected_roas}x`;

    // Low-CLV
    document.getElementById('lowClvBudget').textContent = formatCurrency(allocation['Low-CLV'].budget);
    document.getElementById('lowClvAcq').textContent = formatNumber(allocation['Low-CLV'].expected_acquisitions);
    document.getElementById('lowClvRoas').textContent = `${allocation['Low-CLV'].expected_roas}x`;
}

// Render Budget Chart
function renderBudgetChart(data) {
    if (charts.budget) {
        charts.budget.destroy();
    }

    const ctx = document.getElementById('budgetChart').getContext('2d');
    const allocation = data.allocation;

    charts.budget = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['High-CLV (50%)', 'Growth-Potential (35%)', 'Low-CLV (15%)'],
            datasets: [{
                data: [
                    allocation['High-CLV'].budget,
                    allocation['Growth-Potential'].budget,
                    allocation['Low-CLV'].budget
                ],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(99, 102, 241, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#9ca3af',
                        padding: 20
                    }
                }
            }
        }
    });
}

// Submit Prediction
async function submitPrediction() {
    const formData = {
        total_orders: parseInt(document.getElementById('totalOrders').value),
        total_spent: parseFloat(document.getElementById('totalSpent').value),
        avg_order_value: parseFloat(document.getElementById('avgOrderValue').value),
        days_since_first_purchase: parseInt(document.getElementById('daysSinceFirst').value),
        days_since_last_purchase: parseInt(document.getElementById('daysSinceLast').value),
        num_categories: parseInt(document.getElementById('numCategories').value),
        acquisition_source: document.getElementById('acquisitionSource').value,
        campaign_type: document.getElementById('campaignType').value,
        acquisition_cost: parseFloat(document.getElementById('acquisitionCost').value),
        email_engagement_rate: parseFloat(document.getElementById('emailEngagement').value),
        return_rate: parseFloat(document.getElementById('returnRate').value)
    };

    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        let result;
        if (!response.ok) {
            // Generate demo prediction
            result = generateDemoPrediction(formData);
        } else {
            result = await response.json();
        }

        displayPredictionResult(result);

    } catch (error) {
        console.error('Prediction failed:', error);
        const result = generateDemoPrediction(formData);
        displayPredictionResult(result);
    }
}

// Display Prediction Result
function displayPredictionResult(result) {
    const resultDiv = document.getElementById('predictionResult');
    resultDiv.classList.remove('hidden');

    document.getElementById('resultClv').textContent = formatCurrency(result.predicted_clv);
    document.getElementById('resultSegment').textContent = result.segment;
    document.getElementById('resultSegment').className = `result-value segment-badge ${getSegmentClass(result.segment)}`;
    document.getElementById('resultConfidence').textContent = result.confidence;
    document.getElementById('resultCac').textContent = formatCurrency(result.recommended_cac);

    // Scroll to result
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}

// Helper Functions
function formatCurrency(value) {
    if (value === null || value === undefined) return '$0';
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(value);
}

function formatNumber(value) {
    if (value === null || value === undefined) return '0';
    return new Intl.NumberFormat('en-US').format(value);
}

function getSegmentClass(segment) {
    switch (segment) {
        case 'High-CLV': return 'high-clv';
        case 'Growth-Potential': return 'growth';
        case 'Low-CLV': return 'low-clv';
        default: return '';
    }
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Demo Data Functions (for when API is not available)
function getDemoData() {
    return {
        summary: {
            total_customers: 1000,
            avg_clv: 342.50,
            total_value: 342500,
            high_clv_percentage: 20
        },
        segments: {
            'High-CLV': 200,
            'Growth-Potential': 600,
            'Low-CLV': 200
        },
        acquisition_sources: {
            'Meta Ads': 350,
            'Google Ads': 250,
            'Email': 150,
            'Organic': 100,
            'Direct': 100,
            'Referral': 50
        },
        clv_distribution: {
            '$0-100': 150,
            '$100-250': 250,
            '$250-500': 300,
            '$500-1000': 200,
            '$1000+': 100
        },
        top_customers: [
            { customer_id: 'CUST_00001', predicted_clv: 1850, predicted_segment: 'High-CLV', total_orders: 15, acquisition_source: 'Meta Ads' },
            { customer_id: 'CUST_00002', predicted_clv: 1620, predicted_segment: 'High-CLV', total_orders: 12, acquisition_source: 'Google Ads' },
            { customer_id: 'CUST_00003', predicted_clv: 1450, predicted_segment: 'High-CLV', total_orders: 10, acquisition_source: 'Email' },
            { customer_id: 'CUST_00004', predicted_clv: 1380, predicted_segment: 'High-CLV', total_orders: 9, acquisition_source: 'Organic' },
            { customer_id: 'CUST_00005', predicted_clv: 1250, predicted_segment: 'High-CLV', total_orders: 8, acquisition_source: 'Meta Ads' }
        ]
    };
}

function getDemoCustomers() {
    const customers = [];
    const sources = ['Meta Ads', 'Google Ads', 'Email', 'Direct', 'Organic', 'Referral'];
    const segments = ['High-CLV', 'Growth-Potential', 'Low-CLV'];

    for (let i = 1; i <= 50; i++) {
        customers.push({
            customer_id: `CUST_${String(i).padStart(5, '0')}`,
            total_spent: Math.floor(Math.random() * 2000) + 100,
            predicted_clv: Math.floor(Math.random() * 1500) + 50,
            total_orders: Math.floor(Math.random() * 20) + 1,
            customer_segment: segments[Math.floor(Math.random() * segments.length)],
            acquisition_source: sources[Math.floor(Math.random() * sources.length)],
            email_engagement_rate: Math.random()
        });
    }

    return customers;
}

function getDemoSegments() {
    return {
        segments: {
            'High-CLV': { count: 200, avg_clv: 850, total_value: 170000 },
            'Growth-Potential': { count: 600, avg_clv: 280, total_value: 168000 },
            'Low-CLV': { count: 200, avg_clv: 85, total_value: 17000 }
        },
        total_customers: 1000
    };
}

function getDemoFeatureImportance() {
    return [
        { feature: 'total_spent', importance: 0.25 },
        { feature: 'total_orders', importance: 0.18 },
        { feature: 'email_engagement_rate', importance: 0.15 },
        { feature: 'days_since_last_purchase', importance: 0.12 },
        { feature: 'avg_order_value', importance: 0.10 },
        { feature: 'num_categories', importance: 0.08 },
        { feature: 'acquisition_cost', importance: 0.06 },
        { feature: 'return_rate', importance: 0.04 },
        { feature: 'purchase_velocity', importance: 0.02 }
    ];
}

function getDemoBudgetAllocation(totalBudget) {
    return {
        total_budget: totalBudget,
        allocation: {
            'High-CLV': {
                budget: totalBudget * 0.5,
                budget_percentage: 50,
                expected_acquisitions: Math.floor(totalBudget * 0.5 / 150),
                expected_roas: 3.33
            },
            'Growth-Potential': {
                budget: totalBudget * 0.35,
                budget_percentage: 35,
                expected_acquisitions: Math.floor(totalBudget * 0.35 / 60),
                expected_roas: 2.5
            },
            'Low-CLV': {
                budget: totalBudget * 0.15,
                budget_percentage: 15,
                expected_acquisitions: Math.floor(totalBudget * 0.15 / 25),
                expected_roas: 1.5
            }
        },
        summary: {
            total_expected_acquisitions: Math.floor(totalBudget / 50),
            total_expected_revenue: totalBudget * 2.5,
            blended_roas: 2.5
        }
    };
}

function generateDemoPrediction(formData) {
    // Simple prediction logic based on form data
    let baseCLV = formData.total_spent * 1.5;

    // Adjust based on engagement
    baseCLV *= (1 + formData.email_engagement_rate);

    // Adjust based on return rate
    baseCLV *= (1 - formData.return_rate);

    // Adjust based on orders
    baseCLV *= (1 + formData.total_orders * 0.05);

    // Adjust based on recency
    if (formData.days_since_last_purchase < 30) {
        baseCLV *= 1.2;
    } else if (formData.days_since_last_purchase > 90) {
        baseCLV *= 0.8;
    }

    const predictedCLV = Math.round(baseCLV);

    let segment, confidence;
    if (predictedCLV >= 500) {
        segment = 'High-CLV';
        confidence = 'High';
    } else if (predictedCLV >= 150) {
        segment = 'Growth-Potential';
        confidence = 'Medium';
    } else {
        segment = 'Low-CLV';
        confidence = 'Low';
    }

    return {
        predicted_clv: predictedCLV,
        segment: segment,
        confidence: confidence,
        recommended_cac: Math.round(predictedCLV * 0.3)
    };
}

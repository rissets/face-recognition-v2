<template>
  <div class="api-usage-panel">
    <div class="panel-header">
      <h3>API Usage Statistics</h3>
      <div class="controls">
        <select v-model="selectedEndpoint" @change="loadUsage">
          <option value="">All Endpoints</option>
          <option value="enrollment">Enrollment</option>
          <option value="recognition">Recognition</option>
          <option value="liveness">Liveness</option>
          <option value="analytics">Analytics</option>
          <option value="webhook">Webhook</option>
        </select>
        <button @click="loadUsage" class="refresh-btn">
          <span>🔄</span> Refresh
        </button>
      </div>
    </div>

    <div class="usage-stats" v-if="stats">
      <div class="stat-card">
        <h4>Total API Calls</h4>
        <div class="stat-value">{{ stats.total_calls }}</div>
      </div>
      <div class="stat-card">
        <h4>Success Rate</h4>
        <div class="stat-value success">{{ stats.success_rate }}%</div>
      </div>
      <div class="stat-card">
        <h4>Avg Response Time</h4>
        <div class="stat-value">{{ stats.avg_response_time }}ms</div>
      </div>
      <div class="stat-card">
        <h4>Most Used Endpoint</h4>
        <div class="stat-value">{{ stats.most_used_endpoint }}</div>
      </div>
    </div>

    <div class="usage-chart" v-if="usageData.length > 0">
      <h4>API Usage by Endpoint</h4>
      <div class="chart-container">
        <div 
          v-for="endpoint in endpointStats" 
          :key="endpoint.name"
          class="chart-bar"
        >
          <div 
            class="bar" 
            :style="{ height: `${(endpoint.count / maxCount) * 100}%` }"
            :class="endpoint.name"
          ></div>
          <div class="bar-label">{{ endpoint.name }}</div>
          <div class="bar-count">{{ endpoint.count }}</div>
        </div>
      </div>
    </div>

    <div class="usage-table" v-if="usageData.length > 0">
      <h4>Recent API Calls</h4>
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Endpoint</th>
            <th>Method</th>
            <th>Status</th>
            <th>Response Time</th>
            <th>IP Address</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="usage in usageData.slice(0, 20)" :key="usage.id">
            <td>{{ formatDateTime(usage.created_at) }}</td>
            <td>
              <span class="endpoint-badge" :class="usage.endpoint">
                {{ usage.endpoint }}
              </span>
            </td>
            <td>
              <span class="method-badge" :class="usage.method.toLowerCase()">
                {{ usage.method }}
              </span>
            </td>
            <td>
              <span class="status-badge" :class="getStatusClass(usage.status_code)">
                {{ usage.status_code }}
              </span>
            </td>
            <td>{{ usage.response_time_ms?.toFixed(2) || '0' }}ms</td>
            <td>{{ usage.ip_address }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="loading" v-if="loading">
      Loading API usage data...
    </div>

    <div class="empty-state" v-if="!loading && usageData.length === 0">
      <p>No API usage data available yet.</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { clientApi } from '../services/api'

const loading = ref(false)
const usageData = ref([])
const selectedEndpoint = ref('')
const stats = ref(null)

const endpointStats = computed(() => {
  if (!usageData.value.length) return []
  
  const counts = {}
  usageData.value.forEach(usage => {
    counts[usage.endpoint] = (counts[usage.endpoint] || 0) + 1
  })
  
  return Object.entries(counts)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
})

const maxCount = computed(() => {
  return Math.max(...endpointStats.value.map(s => s.count), 1)
})

async function loadUsage() {
  loading.value = true
  try {
    const params = {}
    if (selectedEndpoint.value) {
      params.endpoint = selectedEndpoint.value
    }
    
    const response = await clientApi.apiUsage(params)
    usageData.value = response.data.results || response.data
    calculateStats()
  } catch (error) {
    console.error('Failed to load API usage:', error)
  } finally {
    loading.value = false
  }
}

function calculateStats() {
  if (!usageData.value.length) {
    stats.value = null
    return
  }

  const total = usageData.value.length
  const successful = usageData.value.filter(u => u.status_code >= 200 && u.status_code < 400).length
  const totalResponseTime = usageData.value.reduce((sum, u) => sum + (u.response_time_ms || 0), 0)
  
  const endpointCounts = {}
  usageData.value.forEach(usage => {
    endpointCounts[usage.endpoint] = (endpointCounts[usage.endpoint] || 0) + 1
  })
  
  const mostUsed = Object.entries(endpointCounts)
    .sort((a, b) => b[1] - a[1])[0]

  stats.value = {
    total_calls: total,
    success_rate: total > 0 ? ((successful / total) * 100).toFixed(1) : 0,
    avg_response_time: total > 0 ? (totalResponseTime / total).toFixed(2) : 0,
    most_used_endpoint: mostUsed ? mostUsed[0] : 'N/A'
  }
}

function formatDateTime(dateStr) {
  return new Date(dateStr).toLocaleString()
}

function getStatusClass(statusCode) {
  if (statusCode >= 200 && statusCode < 300) return 'success'
  if (statusCode >= 300 && statusCode < 400) return 'redirect'
  if (statusCode >= 400 && statusCode < 500) return 'client-error'
  if (statusCode >= 500) return 'server-error'
  return 'unknown'
}

onMounted(() => {
  loadUsage()
})
</script>

<style scoped>
.api-usage-panel {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  border-bottom: 1px solid #eee;
  padding-bottom: 15px;
}

.controls {
  display: flex;
  gap: 10px;
  align-items: center;
}

.controls select {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}

.refresh-btn {
  padding: 8px 12px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.refresh-btn:hover {
  background: #0056b3;
}

.usage-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 25px;
}

.stat-card {
  background: #f8f9fa;
  padding: 15px;
  border-radius: 6px;
  text-align: center;
}

.stat-card h4 {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: #666;
  font-weight: 500;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #333;
}

.stat-value.success {
  color: #28a745;
}

.usage-chart {
  margin-bottom: 25px;
}

.chart-container {
  display: flex;
  align-items: end;
  justify-content: space-around;
  height: 150px;
  background: #f8f9fa;
  padding: 20px;
  border-radius: 6px;
}

.chart-bar {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
  max-width: 80px;
}

.bar {
  width: 30px;
  background: #007bff;
  border-radius: 4px 4px 0 0;
  min-height: 5px;
  margin-bottom: 5px;
}

.bar.enrollment { background: #28a745; }
.bar.recognition { background: #17a2b8; }
.bar.liveness { background: #ffc107; }
.bar.analytics { background: #6f42c1; }
.bar.webhook { background: #fd7e14; }

.bar-label {
  font-size: 12px;
  color: #666;
  margin-bottom: 2px;
}

.bar-count {
  font-size: 12px;
  font-weight: bold;
  color: #333;
}

.usage-table {
  overflow-x: auto;
}

.usage-table table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.usage-table th,
.usage-table td {
  padding: 12px 8px;
  text-align: left;
  border-bottom: 1px solid #eee;
}

.usage-table th {
  background: #f8f9fa;
  font-weight: 600;
  color: #555;
}

.endpoint-badge,
.method-badge,
.status-badge {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

.endpoint-badge.enrollment { background: #d4edda; color: #155724; }
.endpoint-badge.recognition { background: #d1ecf1; color: #0c5460; }
.endpoint-badge.liveness { background: #fff3cd; color: #856404; }
.endpoint-badge.analytics { background: #e2e3f0; color: #383d41; }
.endpoint-badge.webhook { background: #f8d7da; color: #721c24; }

.method-badge.get { background: #d4edda; color: #155724; }
.method-badge.post { background: #cce5ff; color: #004085; }
.method-badge.put { background: #fff3cd; color: #856404; }
.method-badge.delete { background: #f8d7da; color: #721c24; }

.status-badge.success { background: #d4edda; color: #155724; }
.status-badge.redirect { background: #fff3cd; color: #856404; }
.status-badge.client-error { background: #f8d7da; color: #721c24; }
.status-badge.server-error { background: #f5c6cb; color: #721c24; }

.loading,
.empty-state {
  text-align: center;
  padding: 40px;
  color: #666;
}
</style>
import axios from 'axios'

// Update this URL when deploying to Render
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const analyzeCharts = async (chart1, chart2) => {
  const formData = new FormData()
  formData.append('chart_1day', chart1)
  formData.append('chart_1year', chart2)

  const response = await api.post('/api/analyze-charts', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export const analyzeLiveStock = async (symbol) => {
  const response = await api.post('/api/live-stock', { symbol })
  return response.data
}

export default api

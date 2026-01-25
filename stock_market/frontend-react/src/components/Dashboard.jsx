import { useState } from 'react'
import ChartAnalysis from './ChartAnalysis'
import LiveStockAnalysis from './LiveStockAnalysis'
import ResultSummary from './ResultSummary'
import { analyzeCharts, analyzeLiveStock } from '../services/api'

const Dashboard = () => {
  const [chartResult, setChartResult] = useState(null)
  const [stockResult, setStockResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleChartAnalysis = async (chart1, chart2) => {
    setLoading(true)
    try {
      const result = await analyzeCharts(chart1, chart2)
      setChartResult(result)
    } catch (error) {
      alert('Error analyzing charts: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const handleStockAnalysis = async (symbol) => {
    setLoading(true)
    try {
      const result = await analyzeLiveStock(symbol)
      setStockResult(result)
    } catch (error) {
      alert('Error analyzing stock: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <header className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">
          AI Stock Analysis Platform
        </h1>
        <p className="text-gray-400">Educational Technical Analysis Tool</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <ChartAnalysis onAnalyze={handleChartAnalysis} loading={loading} />
        <LiveStockAnalysis onAnalyze={handleStockAnalysis} loading={loading} />
      </div>

      {(chartResult || stockResult) && (
        <ResultSummary chartResult={chartResult} stockResult={stockResult} />
      )}
    </div>
  )
}

export default Dashboard

import { useState } from 'react'

const ChartAnalysis = ({ onAnalyze, loading }) => {
  const [chart1, setChart1] = useState(null)
  const [chart2, setChart2] = useState(null)
  const [preview1, setPreview1] = useState(null)
  const [preview2, setPreview2] = useState(null)

  const handleFile1 = (e) => {
    const file = e.target.files[0]
    if (file) {
      setChart1(file)
      const reader = new FileReader()
      reader.onloadend = () => setPreview1(reader.result)
      reader.readAsDataURL(file)
    }
  }

  const handleFile2 = (e) => {
    const file = e.target.files[0]
    if (file) {
      setChart2(file)
      const reader = new FileReader()
      reader.onloadend = () => setPreview2(reader.result)
      reader.readAsDataURL(file)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (chart1 && chart2) {
      onAnalyze(chart1, chart2)
    } else {
      alert('Please upload both chart images')
    }
  }

  return (
    <div className="bg-trading-card border border-trading-border rounded-lg p-6">
      <h2 className="text-2xl font-semibold text-white mb-4">
        Chart Pattern Analysis
      </h2>
      <p className="text-gray-400 text-sm mb-6">
        Upload two chart images: 1-day/intraday chart and 1-year chart
      </p>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Image 1: 1-Day or Intraday Chart
          </label>
          <input
            type="file"
            accept="image/*"
            onChange={handleFile1}
            className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-trading-green file:text-trading-dark hover:file:bg-trading-green/80"
          />
          {preview1 && (
            <img
              src={preview1}
              alt="Chart 1 preview"
              className="mt-2 max-h-32 rounded border border-trading-border"
            />
          )}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Image 2: 1-Year Chart
          </label>
          <input
            type="file"
            accept="image/*"
            onChange={handleFile2}
            className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-trading-green file:text-trading-dark hover:file:bg-trading-green/80"
          />
          {preview2 && (
            <img
              src={preview2}
              alt="Chart 2 preview"
              className="mt-2 max-h-32 rounded border border-trading-border"
            />
          )}
        </div>

        <button
          type="submit"
          disabled={loading || !chart1 || !chart2}
          className="w-full bg-trading-green text-trading-dark font-semibold py-2 px-4 rounded hover:bg-trading-green/80 disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          {loading ? 'Analyzing...' : 'Analyze Charts'}
        </button>
      </form>
    </div>
  )
}

export default ChartAnalysis

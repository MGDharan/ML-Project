import { useState } from 'react'

const LiveStockAnalysis = ({ onAnalyze, loading }) => {
  const [symbol, setSymbol] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (symbol.trim()) {
      onAnalyze(symbol.trim())
    } else {
      alert('Please enter a stock symbol')
    }
  }

  return (
    <div className="bg-trading-card border border-trading-border rounded-lg p-6">
      <h2 className="text-2xl font-semibold text-white mb-4">
        Live Stock Analysis
      </h2>
      <p className="text-gray-400 text-sm mb-6">
        Enter stock symbol (e.g., TATASTEEL, RELIANCE, INFY)
      </p>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Stock Symbol
          </label>
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="TATASTEEL"
            className="w-full bg-trading-dark border border-trading-border rounded px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-trading-green"
          />
        </div>

        <button
          type="submit"
          disabled={loading || !symbol.trim()}
          className="w-full bg-trading-green text-trading-dark font-semibold py-2 px-4 rounded hover:bg-trading-green/80 disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          {loading ? 'Analyzing...' : 'Analyze Stock'}
        </button>
      </form>

      <div className="mt-4 p-3 bg-trading-dark rounded text-xs text-gray-400">
        <p>Note: For NSE stocks, symbol is automatically detected. Use symbol without .NS suffix.</p>
      </div>
    </div>
  )
}

export default LiveStockAnalysis

import React from 'react'

const ResultSummary = ({ chartResult, stockResult }) => {
  const getTrendBadge = (trend) => {
    const colors = {
      Bullish: 'bg-green-500/20 text-green-400 border-green-500/50',
      Bearish: 'bg-red-500/20 text-red-400 border-red-500/50',
      Sideways: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50'
    }
    return (
      <span className={`px-3 py-1 rounded-full text-sm font-semibold border ${colors[trend] || colors.Sideways}`}>
        {trend}
      </span>
    )
  }

  const getOptionBadge = (bias) => {
    if (bias === 'CE') {
      return <span className="px-3 py-1 rounded-full text-sm font-semibold bg-green-500/20 text-green-400 border border-green-500/50">CALL (CE)</span>
    } else if (bias === 'PE') {
      return <span className="px-3 py-1 rounded-full text-sm font-semibold bg-red-500/20 text-red-400 border border-red-500/50">PUT (PE)</span>
    } else {
      return <span className="px-3 py-1 rounded-full text-sm font-semibold bg-gray-500/20 text-gray-400 border border-gray-500/50">No Trade</span>
    }
  }

  const getRiskBadge = (risk) => {
    const colors = {
      High: 'bg-red-500/20 text-red-400',
      Medium: 'bg-yellow-500/20 text-yellow-400',
      Low: 'bg-green-500/20 text-green-400'
    }
    return (
      <span className={`px-3 py-1 rounded-full text-sm font-semibold ${colors[risk] || colors.Medium}`}>
        {risk} Risk
      </span>
    )
  }

  return (
    <div className="mt-6 space-y-6">
      {chartResult && (
        <div className="bg-trading-card border border-trading-border rounded-lg p-6">
          <h2 className="text-2xl font-semibold text-white mb-4">
            Chart Analysis Results
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <p className="text-gray-400 text-sm mb-1">Detected Pattern</p>
              <p className="text-white font-semibold">{chartResult.detected_pattern}</p>
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-1">Trend</p>
              {getTrendBadge(chartResult.trend)}
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-1">Option Bias</p>
              {getOptionBadge(chartResult.option_bias)}
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-1">Risk Level</p>
              {getRiskBadge(chartResult.risk_level)}
            </div>
          </div>

          <div className="space-y-3 mt-4">
            <div>
              <p className="text-gray-400 text-sm mb-1">Buy Zone</p>
              <p className="text-white">{chartResult.buy_zone}</p>
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-1">Sell Zone</p>
              <p className="text-white">{chartResult.sell_zone}</p>
            </div>
            {chartResult.confidence && (
              <div>
                <p className="text-gray-400 text-sm mb-1">Confidence</p>
                <div className="w-full bg-trading-dark rounded-full h-2">
                  <div
                    className="bg-trading-green h-2 rounded-full"
                    style={{ width: `${chartResult.confidence * 100}%` }}
                  ></div>
                </div>
                <p className="text-gray-400 text-xs mt-1">
                  {(chartResult.confidence * 100).toFixed(1)}%
                </p>
              </div>
            )}
          </div>

          <div className="mt-4 p-3 bg-yellow-900/20 border border-yellow-500/30 rounded">
            <p className="text-yellow-400 text-xs">{chartResult.disclaimer}</p>
          </div>
        </div>
      )}

      {stockResult && (
        <div className="bg-trading-card border border-trading-border rounded-lg p-6">
          <h2 className="text-2xl font-semibold text-white mb-4">
            Live Stock Analysis Results
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <p className="text-gray-400 text-sm mb-1">Symbol</p>
              <p className="text-white font-semibold text-xl">{stockResult.symbol}</p>
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-1">Current Price</p>
              <p className="text-white font-semibold text-xl">Rs. {stockResult.current_price}</p>
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-1">Trend</p>
              {getTrendBadge(stockResult.trend)}
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-1">Momentum</p>
              <p className="text-white font-semibold">{stockResult.momentum}</p>
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-1">Option Bias</p>
              {getOptionBadge(stockResult.option_bias)}
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-1">Risk Level</p>
              {getRiskBadge(stockResult.risk_level)}
            </div>
          </div>

          <div className="mt-4 p-3 bg-yellow-900/20 border border-yellow-500/30 rounded">
            <p className="text-yellow-400 text-xs">{stockResult.disclaimer}</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default ResultSummary

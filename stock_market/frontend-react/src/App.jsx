import { useState } from 'react'
import Dashboard from './components/Dashboard'
import Disclaimer from './components/Disclaimer'
import './App.css'

function App() {
  return (
    <div className="min-h-screen bg-trading-dark">
      <Disclaimer />
      <Dashboard />
    </div>
  )
}

export default App

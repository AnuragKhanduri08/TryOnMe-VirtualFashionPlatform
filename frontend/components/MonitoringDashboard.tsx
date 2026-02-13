"use client"

import { useState, useEffect } from "react"
import axios from "axios"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Activity, AlertCircle, Clock, Server, RefreshCw } from "lucide-react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { Button } from "@/components/ui/button"

export default function MonitoringDashboard() {
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())

  const fetchStats = async () => {
    // Don't set full loading state on refresh to avoid flickering
    // setLoading(true) 
    try {
      const rawUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
      const API_URL = rawUrl.trim()
      const res = await axios.get(`${API_URL}/dashboard/stats`)
      setStats(res.data)
      setLastUpdated(new Date())
    } catch (error) {
      console.error("Failed to fetch stats", error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStats()
    const interval = setInterval(fetchStats, 5000) // Auto refresh every 5s
    return () => clearInterval(interval)
  }, [])

  if (loading && !stats) return <div className="p-8 text-center text-gray-500">Initializing Dashboard...</div>
  if (!stats) return <div className="p-8 text-center text-red-500">Failed to load dashboard data. Is the backend running?</div>

  // Prepare data for charts
  const endpointData = Object.keys(stats.endpoint_usage || {}).map(key => ({
    name: key,
    value: stats.endpoint_usage[key]
  }))

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold tracking-tight">System Monitor</h2>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span>Last updated: {lastUpdated.toLocaleTimeString()}</span>
          <Button size="icon" variant="ghost" onClick={fetchStats}>
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Requests</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.total_requests}</div>
            <p className="text-xs text-muted-foreground">Lifetime API calls</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Latency</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(stats.avg_latency * 1000).toFixed(1)}ms</div>
            <p className="text-xs text-muted-foreground">Last 100 requests</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Error Rate</CardTitle>
            <AlertCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600 dark:text-red-400">{stats.errors}</div>
            <p className="text-xs text-muted-foreground">Total failed requests (5xx/4xx)</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Endpoint Usage Chart */}
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Endpoint Usage</CardTitle>
            <CardDescription>Distribution of API calls by route</CardDescription>
          </CardHeader>
          <CardContent className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={endpointData} layout="vertical" margin={{ left: 20, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="var(--border)" />
                <XAxis type="number" allowDecimals={false} stroke="var(--muted-foreground)" />
                <YAxis dataKey="name" type="category" width={150} tick={{fontSize: 11, fill: 'var(--muted-foreground)'}} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'var(--popover)', borderRadius: '8px', border: '1px solid var(--border)', color: 'var(--popover-foreground)' }}
                  cursor={{fill: 'var(--muted)'}}
                />
                <Bar dataKey="value" fill="var(--primary)" radius={[0, 4, 4, 0]}>
                    {endpointData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={index % 2 === 0 ? 'var(--primary)' : 'var(--primary)'} opacity={index % 2 === 0 ? 1 : 0.8} />
                    ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Recent Logs */}
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>Latest 20 API requests</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4 max-h-[300px] overflow-auto pr-2 custom-scrollbar">
                {stats.recent_logs?.length === 0 && <p className="text-sm text-muted-foreground text-center py-8">No activity yet.</p>}
                {stats.recent_logs?.map((log: any, i: number) => (
                    <div key={i} className="flex items-center justify-between border-b pb-2 last:border-0 last:pb-0 hover:bg-muted/50 p-1 rounded transition-colors">
                        <div className="flex flex-col">
                            <span className="font-medium text-sm flex items-center gap-2">
                                <span className={`w-2 h-2 rounded-full ${log.status < 400 ? 'bg-green-500' : 'bg-destructive'}`}></span>
                                <span className="uppercase text-xs font-bold text-muted-foreground w-10">{log.method}</span>
                                <span className="text-foreground truncate max-w-[150px]" title={log.path}>{log.path}</span>
                            </span>
                            <span className="text-[10px] text-muted-foreground pl-4">
                                {new Date(log.timestamp * 1000).toLocaleTimeString()}
                            </span>
                        </div>
                        <div className="text-right">
                             <span className={`text-xs font-mono px-2 py-0.5 rounded ${log.status < 400 ? 'bg-green-500/10 text-green-600 dark:text-green-400' : 'bg-destructive/10 text-destructive'}`}>
                                {log.status}
                             </span>
                             <div className="text-[10px] text-muted-foreground mt-1">{log.latency}</div>
                        </div>
                    </div>
                ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

import SmartSearch from "@/components/SmartSearch";
import BodyMeasurement from "@/components/BodyMeasurement";
import MonitoringDashboard from "@/components/MonitoringDashboard";
import VirtualTryOn from "@/components/VirtualTryOn";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ModeToggle } from "@/components/theme-toggle";

export default function Home() {
  return (
    <main className="relative flex min-h-screen flex-col items-center p-8 bg-background text-foreground transition-colors duration-300">
      <div className="absolute top-4 right-4 z-50">
        <ModeToggle />
      </div>
      
      <div className="z-10 w-full max-w-5xl flex flex-col items-center justify-center font-mono text-sm mb-8 text-center">
        <h1 className="text-4xl font-bold tracking-tight">Virtual Fashion Platform</h1>
        <p className="text-muted-foreground mt-2">AI-Powered Shopping Experience</p>
      </div>

      <div className="w-full max-w-6xl">
        <Tabs defaultValue="search" className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8">
            <TabsTrigger value="search">Smart Search</TabsTrigger>
            <TabsTrigger value="tryon">Virtual Try-On</TabsTrigger>
            <TabsTrigger value="measure">Body Measurement</TabsTrigger>
            <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
          </TabsList>
          
          <TabsContent value="search">
            <SmartSearch />
          </TabsContent>
          
          <TabsContent value="tryon">
            <VirtualTryOn />
          </TabsContent>
          
          <TabsContent value="measure">
            <BodyMeasurement />
          </TabsContent>

          <TabsContent value="dashboard">
            <MonitoringDashboard />
          </TabsContent>
        </Tabs>
      </div>
    </main>
  );
}

"use client"

import { useState } from "react"
import axios from "axios"
import { Upload, Loader2, Ruler } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import CameraCapture from "./CameraCapture"

export default function BodyMeasurement() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [userHeight, setUserHeight] = useState<string>("170")

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleImageSelection(file)
    }
  }

  const handleImageSelection = (file: File) => {
      setImageFile(file)
      setPreview(URL.createObjectURL(file))
      setResult(null)
  }

  const handleMeasurement = async () => {
    if (!imageFile) return
    setLoading(true)
    const formData = new FormData()
    formData.append("file", imageFile)

    try {
      const res = await axios.post("http://localhost:8000/measure", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      })
      setResult(res.data)
    } catch (error) {
      console.error("Measurement failed", error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>AI Body Measurement</CardTitle>
        <CardDescription>Upload a full-body photo to get estimated measurements.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col gap-4">
          <CameraCapture onCapture={handleImageSelection} />
          
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t" />
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-background px-2 text-muted-foreground">Or upload file</span>
            </div>
          </div>

          <Input 
            type="file" 
            accept="image/*" 
            onChange={handleFileChange} 
          />

          <div className="space-y-2">
            <Label htmlFor="height">Your Height (cm)</Label>
            <Input 
                id="height"
                type="number" 
                value={userHeight} 
                onChange={(e) => setUserHeight(e.target.value)}
                placeholder="e.g. 170"
            />
            <p className="text-xs text-muted-foreground">Used to calibrate pixel measurements to cm.</p>
          </div>
          
          {preview && (
            <div className="relative w-full h-64 bg-muted rounded-md overflow-hidden">
              <img src={preview} alt="Preview" className="w-full h-full object-contain" />
            </div>
          )}

          <Button onClick={handleMeasurement} disabled={!imageFile || loading} className="w-full">
            {loading ? <Loader2 className="animate-spin mr-2" /> : <Ruler className="mr-2" />}
            Get Measurements
          </Button>
        </div>

        {result && (
          <div className="mt-4 p-4 bg-muted/50 rounded-lg border">
            {result.status === "success" ? (
              <div className="space-y-2">
                <h3 className="font-semibold text-green-600 dark:text-green-500">Measurement Successful</h3>
                
                {(() => {
                    const heightCm = parseFloat(userHeight) || 0;
                    const heightPx = result.measurements.estimated_height_pixels || (result.measurements.torso_length_pixels * 3); // Fallback if backend not updated
                    const pxToCm = (heightCm > 0 && heightPx > 0) ? (heightCm / heightPx) : 0;

                    return (
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div className="p-2 bg-card rounded shadow-sm border">
                            <span className="text-muted-foreground">Shoulder Width</span>
                            <div className="flex flex-col">
                                <span className="font-mono text-lg">{result.measurements.shoulder_width_pixels.toFixed(0)} px</span>
                                {pxToCm > 0 && (
                                    <span className="font-mono text-md text-primary font-bold">
                                        {(result.measurements.shoulder_width_pixels * pxToCm).toFixed(1)} cm
                                    </span>
                                )}
                            </div>
                          </div>
                          <div className="p-2 bg-card rounded shadow-sm border">
                            <span className="text-muted-foreground">Torso Length</span>
                            <div className="flex flex-col">
                                <span className="font-mono text-lg">{result.measurements.torso_length_pixels.toFixed(0)} px</span>
                                {pxToCm > 0 && (
                                    <span className="font-mono text-md text-primary font-bold">
                                        {(result.measurements.torso_length_pixels * pxToCm).toFixed(1)} cm
                                    </span>
                                )}
                            </div>
                          </div>
                          <div className="p-2 bg-card rounded shadow-sm border">
                            <span className="text-muted-foreground">Shoulder/Hip Ratio</span>
                            <p className="font-mono text-lg">{result.measurements.shoulder_hip_ratio.toFixed(2)}</p>
                          </div>
                          {pxToCm > 0 && (
                             <div className="p-2 bg-card rounded shadow-sm border">
                                <span className="text-muted-foreground">Est. Full Height (px)</span>
                                <p className="font-mono text-lg">{heightPx.toFixed(0)} px</p>
                             </div>
                          )}
                        </div>
                    );
                })()}

                <p className="text-xs text-muted-foreground mt-2">
                    * CM values are estimates based on provided height. Ensure full body is visible for accuracy.
                </p>
              </div>
            ) : (
              <div className="text-destructive font-medium">
                Error: {result.message}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

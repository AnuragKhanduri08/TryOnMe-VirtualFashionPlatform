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
    
    // Validate height input
    let heightToSend = "170";
    if (userHeight && !isNaN(parseFloat(userHeight))) {
        heightToSend = userHeight;
    }
    formData.append("height", heightToSend)

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

  // Trigger measurement if height changes AND we already have a result (to live update)
  // But be careful not to loop. We need a "Recalculate" button or just rely on user clicking "Get Measurements"
  // Let's add a visual cue or button if height changes?
  // Or just make the "Get Measurements" button always active.


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

          <div className="flex gap-4 items-end">
            <div className="flex-1 space-y-2">
              <Label htmlFor="height">Your Height (cm)</Label>
              <Input 
                id="height" 
                type="number" 
                placeholder="170" 
                value={userHeight}
                onChange={(e) => setUserHeight(e.target.value)}
              />
            </div>
            <Button 
                onClick={handleMeasurement} 
                disabled={!imageFile || loading}
                className="mb-[2px]"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Calculating...
                </>
              ) : (
                 result ? "Recalculate" : "Get Measurements"
              )}
            </Button>
          </div>
          
          {preview && (
            <div className="relative w-full h-64 bg-muted rounded-md overflow-hidden">
              <img src={preview} alt="Preview" className="w-full h-full object-contain" />
            </div>
          )}
        </div>

        {result && (
          <div className="mt-4 p-4 bg-muted/50 rounded-lg border">
            {result.status === "success" ? (
              <div className="space-y-2">
                <h3 className="font-semibold text-green-600 dark:text-green-500">Measurement Successful</h3>
                
                {(() => {
                    // Check if backend provides direct CM estimates (New Logic)
                    const cmEstimates = result.measurements.cm_estimates;
                    
                    if (cmEstimates) {
                        return (
                            <div className="space-y-4">
                                <div className="p-3 bg-primary/10 border border-primary/20 rounded-lg text-center">
                                    <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Recommended Size</span>
                                    <div className="text-4xl font-extrabold text-primary mt-1">{cmEstimates.suggested_size}</div>
                                    <div className="text-xs text-muted-foreground mt-1">Based on shoulder width ({cmEstimates.shoulder_width_cm} cm)</div>
                                </div>
                                
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                  <div className="p-2 bg-card rounded shadow-sm border">
                                    <span className="text-muted-foreground">Shoulder Width</span>
                                    <div className="flex flex-col">
                                        <span className="font-mono text-lg font-bold">
                                            {cmEstimates.shoulder_width_cm} cm <span className="text-muted-foreground text-sm">({cmEstimates.shoulder_width_in} in)</span>
                                        </span>
                                        <span className="text-xs text-muted-foreground">({result.measurements.shoulder_width_pixels.toFixed(0)} px)</span>
                                    </div>
                                  </div>
                                  <div className="p-2 bg-card rounded shadow-sm border">
                                    <span className="text-muted-foreground">Chest Width</span>
                                    <div className="flex flex-col">
                                        <span className="font-mono text-lg font-bold">
                                            {cmEstimates.chest_width_cm} cm <span className="text-muted-foreground text-sm">({cmEstimates.chest_width_in} in)</span>
                                        </span>
                                        <span className="text-xs text-muted-foreground">({result.measurements.chest_width_pixels.toFixed(0)} px)</span>
                                    </div>
                                  </div>
                                  <div className="p-2 bg-card rounded shadow-sm border">
                                    <span className="text-muted-foreground">Waist Width</span>
                                    <div className="flex flex-col">
                                        <span className="font-mono text-lg font-bold">
                                            {cmEstimates.waist_width_cm} cm <span className="text-muted-foreground text-sm">({cmEstimates.waist_width_in} in)</span>
                                        </span>
                                        <span className="text-xs text-muted-foreground">({result.measurements.waist_width_pixels.toFixed(0)} px)</span>
                                    </div>
                                  </div>
                                  <div className="p-2 bg-card rounded shadow-sm border">
                                    <span className="text-muted-foreground">Torso Length</span>
                                    <div className="flex flex-col">
                                        <span className="font-mono text-lg font-bold">
                                            {cmEstimates.torso_length_cm} cm <span className="text-muted-foreground text-sm">({cmEstimates.torso_length_in} in)</span>
                                        </span>
                                        <span className="text-xs text-muted-foreground">({result.measurements.torso_length_pixels.toFixed(0)} px)</span>
                                    </div>
                                  </div>
                                  <div className="p-2 bg-card rounded shadow-sm border">
                                    <span className="text-muted-foreground">Hip Width</span>
                                    <div className="flex flex-col">
                                        <span className="font-mono text-lg font-bold">
                                            {cmEstimates.hip_width_cm} cm <span className="text-muted-foreground text-sm">({cmEstimates.hip_width_in} in)</span>
                                        </span>
                                        <span className="text-xs text-muted-foreground">({result.measurements.hip_width_pixels.toFixed(0)} px)</span>
                                    </div>
                                  </div>
                                  <div className="p-2 bg-card rounded shadow-sm border">
                                    <span className="text-muted-foreground">Calibration</span>
                                    <div className="flex flex-col">
                                        <span className="font-mono text-lg font-bold text-primary">
                                            {userHeight || "170"} cm
                                        </span>
                                        <span className="text-xs text-muted-foreground">
                                            {cmEstimates.estimated_height_cm 
                                                ? "(Full Body Detected)"
                                                : "(Input Used)"}
                                        </span>
                                    </div>
                                  </div>
                                </div>
                                <div className="text-xs text-muted-foreground text-center italic">
                                    Confidence: {cmEstimates.confidence}
                                </div>
                            </div>
                        )
                    }

                    // Fallback to old frontend calculation if backend outdated
                    const heightCm = parseFloat(userHeight) || 0;
                    const heightPx = result.measurements.estimated_height_pixels || (result.measurements.torso_length_pixels * 3); 
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
                          {/* ... Old Logic ... */}
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

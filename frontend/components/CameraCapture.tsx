"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Camera, StopCircle, Aperture } from "lucide-react"

interface CameraCaptureProps {
  onCapture: (file: File) => void
}

export default function CameraCapture({ onCapture }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "user" } 
      })
      setStream(mediaStream)
      setIsStreaming(true)
    } catch (err) {
      console.error("Error accessing camera:", err)
      alert("Could not access camera. Please ensure you have granted permissions.")
    }
  }

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
      setIsStreaming(false)
    }
  }

  const captureImage = () => {
    if (videoRef.current) {
      const video = videoRef.current
      const canvas = document.createElement("canvas")
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      
      const ctx = canvas.getContext("2d")
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" })
            onCapture(file)
            stopCamera()
          }
        }, "image/jpeg", 0.95)
      }
    }
  }

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream
    }
    
    // Cleanup on unmount
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
    }
  }, [stream])

  return (
    <div className="flex flex-col items-center gap-4 w-full">
      {!isStreaming ? (
        <Button onClick={startCamera} variant="outline" className="w-full">
          <Camera className="mr-2 h-4 w-4" /> Use Camera
        </Button>
      ) : (
        <div className="flex flex-col items-center gap-2 w-full animate-in fade-in zoom-in duration-300">
          <div className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              className="w-full h-full object-cover transform scale-x-[-1]" // Mirror effect
            />
          </div>
          <div className="flex gap-2 w-full">
            <Button onClick={captureImage} className="flex-1" variant="default">
              <Aperture className="mr-2 h-4 w-4" /> Capture
            </Button>
            <Button onClick={stopCamera} variant="destructive" size="icon">
              <StopCircle className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

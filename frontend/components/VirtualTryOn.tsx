"use client"

import { useState } from "react"
import axios from "axios"
import { Upload, Shirt, User, Sparkles, Loader2, Search, Sliders, Scan } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import CameraCapture from "./CameraCapture"

import { Switch } from "@/components/ui/switch"
import { MouseEvent } from "react"
import Cropper, { Point, Area } from 'react-easy-crop'
import { getCroppedImg } from "@/lib/utils"
import Image from "next/image"

interface Product {
    id: string | number
    name: string
    category: string
    image_url: string
}

export default function VirtualTryOn({ initialProduct }: { initialProduct?: Product }) {
  const [personFile, setPersonFile] = useState<File | null>(null)
  const [clothFile, setClothFile] = useState<File | null>(null)
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(initialProduct || null)
  
  const [personPreview, setPersonPreview] = useState<string | null>(null)
  const [clothPreview, setClothPreview] = useState<string | null>(initialProduct?.image_url || null)
  const [resultImage, setResultImage] = useState<string | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [useCloud, setUseCloud] = useState(true) // Default to True for better quality
    const [hfToken, setHfToken] = useState("")
    
    // Manual Mode State
    const [manualMode, setManualMode] = useState(false)
    const [segmentedCloth, setSegmentedCloth] = useState<string | null>(null)
    const [dragPos, setDragPos] = useState({ x: 0, y: 0 })
    const [isDragging, setIsDragging] = useState(false)
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
    
    // Adjustments
    const [scale, setScale] = useState(1.0)
  const [offsetX, setOffsetX] = useState(0)
  const [offsetY, setOffsetY] = useState(0)

  // Advanced Warping State
  const [scaleX, setScaleX] = useState(1.0)
  const [scaleY, setScaleY] = useState(1.0)
  const [rotate, setRotate] = useState(0)
  const [taper, setTaper] = useState(0) // Perspective Rotate X

  // Sleeve Control State
  const [sleeveMode, setSleeveMode] = useState(false)
  const [sleeveImages, setSleeveImages] = useState<{ left: string, torso: string, right: string } | null>(null)
  const [leftSleeveRot, setLeftSleeveRot] = useState(0)
  const [rightSleeveRot, setRightSleeveRot] = useState(0)
  
  // Independent Dimensions
  const [sleeveLength, setSleeveLength] = useState(1.0)
  const [sleeveWidth, setSleeveWidth] = useState(1.0)
  const [torsoScaleX, setTorsoScaleX] = useState(1.0)
  const [torsoScaleY, setTorsoScaleY] = useState(1.0)

  // Product Dialog State
  const [isProductDialogOpen, setIsProductDialogOpen] = useState(false)
  const [products, setProducts] = useState<Product[]>([])
  const [productLoading, setProductLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")

  // Crop State
  const [isCropDialogOpen, setIsCropDialogOpen] = useState(false)
  const [cropImage, setCropImage] = useState<string | null>(null)
  const [crop, setCrop] = useState<Point>({ x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)
  const [croppedAreaPixels, setCroppedAreaPixels] = useState<Area | null>(null)

  const fetchProducts = async (q = "") => {
      setProductLoading(true)
      try {
          const rawUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
          const API_URL = rawUrl.trim()
          // Filter strictly for Topwear to prevent incompatible items
          const url = q 
              ? `${API_URL}/search?q=${q}&limit=20&subCategory=Topwear` 
              : `${API_URL}/products?limit=20&subCategory=Topwear`
          const res = await axios.get(url)
          const data = res.data.results || res.data
          setProducts(Array.isArray(data) ? data : [])
      } catch (e) {
          console.error("Failed to fetch products", e)
      } finally {
          setProductLoading(false)
      }
  }

  const openProductDialog = () => {
      setIsProductDialogOpen(true)
      fetchProducts()
  }

  const selectProduct = (product: Product) => {
      setSelectedProduct(product)
      setClothPreview(product.image_url)
      setClothFile(null) 
      setIsProductDialogOpen(false)
  }

  const handlePersonFile = (file: File) => {
      setPersonFile(file)
      setPersonPreview(URL.createObjectURL(file))
  }


  const handleTryOn = async () => {
    if (!personFile || (!clothFile && !selectedProduct)) {
        setError("Please upload a person image and select a clothing item.")
        return
    }
    
    setLoading(true)
    setError(null)
    setResultImage(null)

    const formData = new FormData()
    formData.append("person_image", personFile)
    
    if (selectedProduct) {
        formData.append("cloth_id", String(selectedProduct.id))
    } else if (clothFile) {
        formData.append("cloth_image", clothFile)
    }

    formData.append("adj_scale", scale.toString())
    formData.append("adj_x", offsetX.toString())
    formData.append("adj_y", offsetY.toString())
    formData.append("use_cloud", useCloud.toString())

    if (useCloud && hfToken) {
        formData.append("hf_token", hfToken)
    }

    try {
      const rawUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
      const API_URL = rawUrl.trim()
      const res = await axios.post(`${API_URL}/try-on`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      })
      setResultImage(res.data.result_image)
      setManualMode(false) // Turn off manual mode if auto is used
    } catch (error) {
      console.error("Try-On failed", error)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      setError((error as any).response?.data?.detail || "Virtual Try-On failed. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  // Helper to split image into 3 parts with overlap to prevent gaps
  const splitSleeves = async (imageUrl: string) => {
      const img = new window.Image()
      img.src = imageUrl
      img.crossOrigin = "anonymous"
      await new Promise(r => img.onload = r)

      const canvas = document.createElement("canvas")
      const ctx = canvas.getContext("2d")
      if (!ctx) return

      const w = img.width
      const h = img.height
      
      // Define Splits with Overlap
      // Left: 0 - 35% (Pivots on right side)
      // Torso: 25% - 75% (Centered)
      // Right: 65% - 100% (Pivots on left side)
      
      const leftEnd = Math.floor(w * 0.35)
      const torsoStart = Math.floor(w * 0.25)
      const torsoEnd = Math.floor(w * 0.75)
      const rightStart = Math.floor(w * 0.65)

      // Left Sleeve
      canvas.width = leftEnd
      canvas.height = h
      ctx.clearRect(0, 0, leftEnd, h)
      ctx.drawImage(img, 0, 0, leftEnd, h, 0, 0, leftEnd, h)
      const leftUrl = canvas.toDataURL()

      // Torso
      const torsoW = torsoEnd - torsoStart
      canvas.width = torsoW
      canvas.height = h
      ctx.clearRect(0, 0, torsoW, h)
      ctx.drawImage(img, torsoStart, 0, torsoW, h, 0, 0, torsoW, h)
      const torsoUrl = canvas.toDataURL()

      // Right Sleeve
      const rightW = w - rightStart
      canvas.width = rightW
      canvas.height = h
      ctx.clearRect(0, 0, rightW, h)
      ctx.drawImage(img, rightStart, 0, rightW, h, 0, 0, rightW, h)
      const rightUrl = canvas.toDataURL()

      setSleeveImages({ left: leftUrl, torso: torsoUrl, right: rightUrl })
      setSleeveMode(true)
  }

  const handleManualInit = async () => {
    if (!clothFile && !selectedProduct) {
        setError("Please select a clothing item first.")
        return
    }
    
    // Open Crop Dialog
    if (selectedProduct) {
        setCropImage(selectedProduct.image_url)
    } else if (clothFile) {
        setCropImage(URL.createObjectURL(clothFile))
    }
    setIsCropDialogOpen(true)
  }

  const onCropComplete = (croppedArea: Area, croppedAreaPixels: Area) => {
    setCroppedAreaPixels(croppedAreaPixels)
  }

  const handleCropSave = async () => {
    if (!cropImage || !croppedAreaPixels) return

    try {
        setLoading(true)
        setIsCropDialogOpen(false)
        setManualMode(true) // Switch view immediately to show loading state
        
        const croppedBlob = await getCroppedImg(cropImage, croppedAreaPixels)
        if (!croppedBlob) {
            throw new Error("Failed to crop image")
        }

        const formData = new FormData()
        formData.append("file", croppedBlob, "cropped_cloth.jpg")
        
        const res = await axios.post("http://localhost:8000/segment", formData, {
            responseType: 'blob'
        })
        const url = URL.createObjectURL(res.data)
        setSegmentedCloth(url)
        // Reset positions
        setDragPos({ x: 0, y: 0 })
        setScale(1.0)
        setScaleX(1.0)
        setScaleY(1.0)
        setRotate(0)
        setTaper(0)
        setSleeveMode(false)
        setSleeveImages(null)
        // Reset independent scales
        setSleeveLength(1.0)
        setSleeveWidth(1.0)
        setTorsoScaleX(1.0)
        setTorsoScaleY(1.0)
    } catch (error) {
        console.error("Segmentation/Crop failed", error)
        setError("Failed to process cloth image. Please try again.")
        setManualMode(false)
    } finally {
        setLoading(false)
    }
  }

  return (
    <div className="w-full max-w-4xl mx-auto space-y-8 animate-in fade-in duration-500">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Virtual Try-On Room</h2>
        <p className="text-gray-500">Upload your photo and a piece of clothing to see how it looks.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Input Section */}
        <Card className="h-full">
            <CardHeader>
                <CardTitle>Upload Images</CardTitle>
                <CardDescription>Select your photo and the item you want to try.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
                <div className="space-y-2">
                    <Label htmlFor="person-upload">1. Your Photo (Full Body/Upper Body)</Label>
                    
                    <div className="mb-4">
                        <CameraCapture onCapture={handlePersonFile} />
                    </div>
                    
                    <div className="relative border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center text-center hover:bg-gray-50 transition">
                        <Input 
                            id="person-upload" 
                            type="file" 
                            accept="image/*" 
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                            onChange={(e) => {
                                const file = e.target.files?.[0]
                                if (file) handlePersonFile(file)
                            }}
                        />
                        {personPreview ? (
                             <div className="relative w-full h-32 z-0">
                                <Image src={personPreview} alt="Person" fill className="object-contain rounded" />
                                <div className="absolute bottom-0 left-0 right-0 bg-white/70 p-1">
                                    <p className="text-xs text-green-600 font-medium truncate">{personFile?.name}</p>
                                </div>
                             </div>
                        ) : (
                            <div className="z-0">
                                <User className="w-8 h-8 text-gray-400 mb-2 mx-auto" />
                                <span className="text-sm text-gray-500">Click to upload your photo</span>
                            </div>
                        )}
                    </div>
                </div>

                <div className="space-y-2">
                    <Label>2. Select Clothing Item</Label>
                    <div 
                        className="relative border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center text-center hover:bg-gray-50 transition cursor-pointer"
                        onClick={openProductDialog}
                    >
                         {clothPreview ? (
                             <div className="relative w-full h-32 z-0">
                                <Image 
                                    src={clothPreview} 
                                    alt="Cloth" 
                                    fill
                                    className="object-contain rounded" 
                                    style={{ imageRendering: "pixelated" }}
                                />
                                <div className="absolute bottom-0 left-0 right-0 bg-white/70 p-1">
                                    <p className="text-xs text-primary font-medium truncate">
                                        {selectedProduct ? selectedProduct.name : clothFile?.name}
                                    </p>
                                </div>
                             </div>
                        ) : (
                            <div className="z-0">
                                <Shirt className="w-8 h-8 text-gray-400 mb-2 mx-auto" />
                                <span className="text-sm text-gray-500">Click to select from catalog</span>
                            </div>
                        )}
                    </div>
                </div>

                {error && (
                    <div className="bg-red-50 text-red-600 text-sm p-3 rounded-md">
                        {error}
                    </div>
                )}

                <div className="flex items-center justify-between space-x-2 border p-3 rounded-lg bg-muted/20">
                    <div className="space-y-0.5">
                        <Label className="text-base">AI Generation Mode (Cloud)</Label>
                        <p className="text-xs text-muted-foreground">
                            Uses IDM-VTON (Slow, High Quality). Off = Standard Mode (Fast, Local Warping).
                        </p>
                    </div>
                    <Switch
                        checked={useCloud}
                        onCheckedChange={setUseCloud}
                    />
                </div>

                {!useCloud && (
                    <div className="bg-yellow-500/10 border border-yellow-500/20 p-3 rounded-lg text-sm text-yellow-600 dark:text-yellow-400">
                        ⚠️ <strong>Note:</strong> Standard Mode (Local) is very basic. It simply overlays the image. For realistic results, please enable <strong>Cloud Mode</strong>.
                    </div>
                )}

                {useCloud && (
                    <div className="space-y-2 border p-3 rounded-lg bg-muted/20">
                         <div className="space-y-0.5">
                            <Label className="text-sm">Hugging Face Token (Optional)</Label>
                            <p className="text-xs text-muted-foreground">
                                Leave empty to use free public servers. If they are busy, adding a token helps.
                            </p>
                        </div>
                        <Input 
                            type="password" 
                            placeholder="hf_..." 
                            value={hfToken}
                            onChange={(e) => setHfToken(e.target.value)}
                            className="bg-background"
                        />
                        <p className="text-[10px] text-muted-foreground">
                            Don&apos;t have a token? <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer" className="underline hover:text-primary">Create one here</a> (Select &apos;Read&apos; role).
                        </p>
                    </div>
                )}

                <div className="grid grid-cols-2 gap-4">
                    <Button onClick={handleTryOn} disabled={loading} className="w-full h-12 text-lg">
                        {loading && !manualMode ? (
                            <>
                                <Loader2 className="animate-spin mr-2" /> 
                                {useCloud ? "Cloud..." : "Auto..."}
                            </>
                        ) : (
                            <>
                                <Sparkles className="mr-2" /> Auto Try-On
                            </>
                        )}
                    </Button>

                    <Button variant="outline" onClick={handleManualInit} disabled={loading} className="w-full h-12 text-lg">
                         {loading && manualMode ? (
                            <Loader2 className="animate-spin mr-2" />
                        ) : (
                            <Sliders className="mr-2" />
                        )}
                        Manual Mode
                    </Button>
                </div>

                {/* Adjustments Section */}
                <div className="pt-4 border-t">
                    <div className="flex items-center gap-2 mb-4 text-sm font-medium text-foreground">
                        <Sliders className="w-4 h-4" />
                        <span>Manual Adjustments</span>
                    </div>
                    
                    <div className="space-y-4">
                        <div className="space-y-2">
                            <div className="flex justify-between text-xs text-muted-foreground">
                                <Label>Size (Scale)</Label>
                                <span>{scale.toFixed(1)}x</span>
                            </div>
                            <Slider 
                                defaultValue={[1.0]} 
                                min={0.5} 
                                max={2.0} 
                                step={0.1} 
                                value={[scale]}
                                onValueChange={(vals) => setScale(vals[0])}
                            />
                        </div>

                        <div className="space-y-2">
                            <div className="flex justify-between text-xs text-muted-foreground">
                                <Label>Vertical Position</Label>
                                <span>{offsetY > 0 ? `+${offsetY}` : offsetY}</span>
                            </div>
                            <Slider 
                                defaultValue={[0]} 
                                min={-0.5} 
                                max={0.5} 
                                step={0.05} 
                                value={[offsetY]}
                                onValueChange={(vals) => setOffsetY(vals[0])}
                            />
                        </div>

                        <div className="space-y-2">
                            <div className="flex justify-between text-xs text-muted-foreground">
                                <Label>Horizontal Position</Label>
                                <span>{offsetX > 0 ? `+${offsetX}` : offsetX}</span>
                            </div>
                            <Slider 
                                defaultValue={[0]} 
                                min={-0.5} 
                                max={0.5} 
                                step={0.05} 
                                value={[offsetX]}
                                onValueChange={(vals) => setOffsetX(vals[0])}
                            />
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>

        {/* Result Section */}
        <Card className="h-full">
            <CardHeader>
                <CardTitle>Result</CardTitle>
                <CardDescription>AI-generated visualization.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                {manualMode ? (
                    <div className="flex flex-col w-full gap-4">
                     <div 
                        className="relative w-full h-[500px] flex items-center justify-center bg-gray-100 overflow-hidden cursor-crosshair rounded-lg border"
                        onMouseMove={(e) => {
                            if (isDragging) {
                                 setDragPos({
                                     x: e.clientX - dragStart.x,
                                     y: e.clientY - dragStart.y
                                 })
                            }
                        }}
                        onMouseUp={() => setIsDragging(false)}
                        onMouseLeave={() => setIsDragging(false)}
                    >
                        {/* Background Person */}
                        {personPreview && (
                            <Image 
                                src={personPreview} 
                                alt="Person Preview"
                                fill
                                className="object-contain pointer-events-none select-none" 
                            />
                        )}
                        
                        {/* Draggable Cloth */}
            {segmentedCloth && !sleeveMode && (
                <div
                    style={{
                                    position: 'absolute',
                                    top: '50%',
                                    left: '50%',
                                    transform: `
                                        translate(-50%, -50%) 
                                        translate(${dragPos.x}px, ${dragPos.y}px) 
                                        perspective(1000px)
                                        rotateX(${taper}deg)
                                        rotate(${rotate}deg)
                                        scale(${scale})
                                        scale(${scaleX}, ${scaleY})
                                    `,
                                    transformOrigin: 'top center', // Rotate from top to allow waist tapering effect
                                    cursor: isDragging ? 'grabbing' : 'grab',
                                    width: '300px', // Base width
                                    pointerEvents: 'auto',
                                    zIndex: 20
                                }}
                                onMouseDown={(e) => {
                                    e.preventDefault()
                                    e.stopPropagation()
                                    setIsDragging(true)
                                    setDragStart({
                                        x: e.clientX - dragPos.x,
                                        y: e.clientY - dragPos.y
                                    })
                                }}
                            >
                                <Image 
                                    src={segmentedCloth}
                                    alt="Cloth Overlay"
                                    width={0}
                                    height={0}
                                    sizes="100vw"
                                    className="w-full h-auto pointer-events-none select-none"
                                />
                            </div>
                        )}

                        {/* Sleeve Mode Cloth */}
                        {sleeveMode && sleeveImages && (
                            <div
                                style={{
                                    position: 'absolute',
                                    top: '50%',
                                    left: '50%',
                                    transform: `
                                        translate(-50%, -50%) 
                                        translate(${dragPos.x}px, ${dragPos.y}px) 
                                        perspective(1000px)
                                        rotateX(${taper}deg)
                                        rotate(${rotate}deg)
                                        scale(${scale})
                                        scale(${scaleX}, ${scaleY})
                                    `,
                                    cursor: isDragging ? 'grabbing' : 'grab',
                                    width: '300px', // Base width
                                    height: '400px', // Explicit height for hit area
                                    pointerEvents: 'auto',
                                    zIndex: 20
                                }}
                                onMouseDown={(e) => {
                                    e.preventDefault()
                                    e.stopPropagation()
                                    setIsDragging(true)
                                    setDragStart({
                                        x: e.clientX - dragPos.x,
                                        y: e.clientY - dragPos.y
                                    })
                                }}
                            >
                                 {/* Left Sleeve */}
                                 <Image 
                                    src={sleeveImages.left}
                                    alt="Left Sleeve"
                                    width={0}
                                    height={0}
                                    sizes="33vw"
                                    style={{
                                        position: 'absolute',
                                        left: '0',
                                        top: '0',
                                        width: '35%',
                                        height: 'auto',
                                        transformOrigin: '100% 15%', // Pivot near shoulder
                                        transform: `rotate(${leftSleeveRot}deg) scale(${sleeveLength}, ${sleeveWidth})`,
                                        zIndex: 1
                                    }}
                                 />
                                 {/* Torso */}
                                 <Image 
                                    src={sleeveImages.torso}
                                    alt="Torso"
                                    width={0}
                                    height={0}
                                    sizes="50vw"
                                    style={{
                                        position: 'absolute',
                                        left: '25%',
                                        top: '0',
                                        width: '50%',
                                        height: 'auto',
                                        transform: `scale(${torsoScaleX}, ${torsoScaleY})`,
                                        zIndex: 2
                                    }}
                                 />
                                 {/* Right Sleeve */}
                                 <Image 
                                    src={sleeveImages.right}
                                    alt="Right Sleeve"
                                    width={0}
                                    height={0}
                                    sizes="33vw"
                                    style={{
                                        position: 'absolute',
                                        left: '65%',
                                        top: '0',
                                        width: '35%',
                                        height: 'auto',
                                        transformOrigin: '0% 15%', // Pivot near shoulder
                                        transform: `rotate(${rightSleeveRot}deg) scale(${sleeveLength}, ${sleeveWidth})`,
                                        zIndex: 1
                                    }}
                                 />
                            </div>
                        )}
                         
                        <div className="absolute top-2 left-2 bg-black/50 text-white p-2 rounded text-xs pointer-events-none z-30">
                            Manual Mode Active: Drag to move, use Slider to resize
                        </div>
                    </div>

                    {/* Manual Controls */}
                    <div className="space-y-4 w-full">
                        <div className="flex items-center justify-between">
                            <h4 className="font-semibold text-sm flex items-center gap-2">
                                 <Sliders className="w-4 h-4" /> Fit Adjustments
                            </h4>
                            {!sleeveMode && segmentedCloth && (
                                <Button 
                                    size="sm" 
                                    variant="outline" 
                                    onClick={() => splitSleeves(segmentedCloth)}
                                    className="h-7 text-xs"
                                >
                                    Enable Sleeve Control
                                </Button>
                            )}
                             {sleeveMode && (
                                <Button 
                                    size="sm" 
                                    variant="destructive" 
                                    onClick={() => {
                                        setSleeveMode(false)
                                        setLeftSleeveRot(0)
                                        setRightSleeveRot(0)
                                        setSleeveLength(1.0)
                                        setSleeveWidth(1.0)
                                        setTorsoScaleX(1.0)
                                        setTorsoScaleY(1.0)
                                    }}
                                    className="h-7 text-xs"
                                >
                                    Reset Sleeves
                                </Button>
                            )}
                        </div>
                        
                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-1">
                                <Label className="text-xs">Overall Size</Label>
                                <Slider value={[scale]} min={0.5} max={2.0} step={0.05} onValueChange={(v) => setScale(v[0])} />
                            </div>
                            <div className="space-y-1">
                                <Label className="text-xs">Waist Taper (Morph)</Label>
                                <Slider value={[taper]} min={-60} max={60} step={1} onValueChange={(v) => setTaper(v[0])} />
                            </div>
                            <div className="space-y-1">
                                <Label className="text-xs">Width Stretch</Label>
                                <Slider value={[scaleX]} min={0.5} max={2.0} step={0.05} onValueChange={(v) => setScaleX(v[0])} />
                            </div>
                            <div className="space-y-1">
                                <Label className="text-xs">Height Stretch</Label>
                                <Slider value={[scaleY]} min={0.5} max={2.0} step={0.05} onValueChange={(v) => setScaleY(v[0])} />
                            </div>
                            <div className="space-y-1">
                                <Label className="text-xs">Rotation</Label>
                                <Slider value={[rotate]} min={-45} max={45} step={1} onValueChange={(v) => setRotate(v[0])} />
                            </div>
                        </div>
                        
                        {sleeveMode && (
                            <div className="grid grid-cols-2 gap-4 mt-4 pt-4 border-t">
                                <div className="space-y-1">
                                    <Label className="text-xs">Left Sleeve Angle</Label>
                                    <Slider value={[leftSleeveRot]} min={-60} max={60} step={1} onValueChange={(v) => setLeftSleeveRot(v[0])} />
                                </div>
                                <div className="space-y-1">
                                    <Label className="text-xs">Right Sleeve Angle</Label>
                                    <Slider value={[rightSleeveRot]} min={-60} max={60} step={1} onValueChange={(v) => setRightSleeveRot(v[0])} />
                                </div>
                                <div className="space-y-1">
                                    <Label className="text-xs">Sleeve Length</Label>
                                    <Slider value={[sleeveLength]} min={0.5} max={2.0} step={0.05} onValueChange={(v) => setSleeveLength(v[0])} />
                                </div>
                                <div className="space-y-1">
                                    <Label className="text-xs">Sleeve Width</Label>
                                    <Slider value={[sleeveWidth]} min={0.5} max={2.0} step={0.05} onValueChange={(v) => setSleeveWidth(v[0])} />
                                </div>
                                <div className="space-y-1">
                                    <Label className="text-xs">Torso Width</Label>
                                    <Slider value={[torsoScaleX]} min={0.5} max={2.0} step={0.05} onValueChange={(v) => setTorsoScaleX(v[0])} />
                                </div>
                                <div className="space-y-1">
                                    <Label className="text-xs">Torso Height</Label>
                                    <Slider value={[torsoScaleY]} min={0.5} max={2.0} step={0.05} onValueChange={(v) => setTorsoScaleY(v[0])} />
                                </div>
                            </div>
                        )}

                        <p className="text-xs text-muted-foreground mt-2">
                            Tip: Use &quot;Waist Taper&quot; to fit the shirt to your body shape.
                        </p>
                    </div>
                    </div>
                ) : (
                    resultImage ? (
                        <Image 
                            src={resultImage} 
                            alt="Try-On Result" 
                            width={0}
                            height={0}
                            sizes="100vw"
                            style={{ width: 'auto', height: 'auto', maxHeight: '500px', maxWidth: '100%' }}
                            className="object-contain rounded-md shadow-lg" 
                        />
                    ) : (
                        <div className="flex flex-col items-center justify-center h-[500px] text-center text-muted-foreground border-2 border-dashed rounded-lg">
                            <Sparkles className="w-12 h-12 mx-auto mb-2 opacity-20" />
                            <p>Result will appear here</p>
                        </div>
                    )
                )}
            </CardContent>
        </Card>
      </div>
      <Dialog open={isProductDialogOpen} onOpenChange={setIsProductDialogOpen}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
                <DialogTitle>Select from Catalog</DialogTitle>
                <DialogDescription>
                    Search and select a clothing item to try on.
                </DialogDescription>
            </DialogHeader>
            
            <div className="space-y-4">
                <div className="relative">
                    <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                    <Input
                        placeholder="Search products..." 
                        className="pl-9" 
                        value={searchQuery}
                        onChange={(e) => {
                            setSearchQuery(e.target.value)
                            if (e.target.value.length > 2) fetchProducts(e.target.value)
                            else if (e.target.value.length === 0) fetchProducts()
                        }}
                    />
                </div>

                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    {productLoading ? (
                        <div className="col-span-full text-center py-8">
                            <Loader2 className="w-8 h-8 animate-spin mx-auto text-primary" />
                        </div>
                    ) : products.length > 0 ? (
                        products.map((p) => (
                            <div 
                                key={p.id} 
                                className="border rounded-lg p-2 cursor-pointer hover:border-primary hover:bg-primary/5 transition group"
                                onClick={() => selectProduct(p)}
                            >
                                <div className="aspect-square relative bg-muted rounded mb-2 overflow-hidden">
                                    <Image 
                                        src={p.image_url} 
                                        alt={p.name} 
                                        fill
                                        className="object-contain mix-blend-multiply dark:mix-blend-normal"
                                        style={{ imageRendering: "pixelated" }}
                                    />
                                </div>
                                <p className="text-xs font-medium truncate">{p.name}</p>
                                <p className="text-xs text-muted-foreground">{p.category}</p>
                            </div>
                        ))
                    ) : (
                        <div className="col-span-full text-center py-8 text-muted-foreground">
                            No products found.
                        </div>
                    )}
                </div>
            </div>
        </DialogContent>
      </Dialog>

      <Dialog open={isCropDialogOpen} onOpenChange={setIsCropDialogOpen}>
        <DialogContent className="max-w-3xl h-[80vh] flex flex-col">
            <DialogHeader>
                <DialogTitle>Crop Clothing Item</DialogTitle>
                <DialogDescription>
                    Crop the image to focus on the clothing item. This helps remove the model&apos;s body.
                </DialogDescription>
            </DialogHeader>
            <div className="flex-1 relative bg-black min-h-[300px]">
                {cropImage && (
                    <Cropper
                        image={cropImage}
                        crop={crop}
                        zoom={zoom}
                        aspect={3 / 4}
                        onCropChange={setCrop}
                        onCropComplete={onCropComplete}
                        onZoomChange={setZoom}
                    />
                )}
            </div>
            <div className="flex items-center justify-between gap-4 py-4">
                <div className="flex items-center gap-2 flex-1">
                    <Label>Zoom</Label>
                    <Slider 
                        value={[zoom]} 
                        min={1} 
                        max={3} 
                        step={0.1} 
                        onValueChange={(vals) => setZoom(vals[0])}
                    />
                </div>
                <div className="flex gap-2">
                    <Button variant="outline" onClick={() => setIsCropDialogOpen(false)}>Cancel</Button>
                    <Button onClick={handleCropSave}>Confirm & Segment</Button>
                </div>
            </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

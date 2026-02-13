"use client"

import { useState } from "react"
import axios from "axios"
import { Search, Upload, Loader2, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"

export default function SmartSearch() {
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [imageFile, setImageFile] = useState<File | null>(null)
  
  // Recommendation State
  const [selectedProduct, setSelectedProduct] = useState<any>(null)
  const [recommendations, setRecommendations] = useState<any[]>([])
  const [matchingItems, setMatchingItems] = useState<any[]>([])
  const [recLoading, setRecLoading] = useState(false)
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  
  // Autocomplete State
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  
  // Product Detail State
  const [detailProduct, setDetailProduct] = useState<any>(null)
  const [isDetailOpen, setIsDetailOpen] = useState(false)

  const fetchSuggestions = async (value: string) => {
    if (!value || value.length < 2) {
        setSuggestions([])
        return
    }
    try {
        const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
        const res = await axios.get(`${API_URL}/search/suggestions?q=${value}`)
        setSuggestions(res.data.suggestions)
        setShowSuggestions(true)
    } catch (error) {
        console.error("Failed to fetch suggestions", error)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value
      setQuery(value)
      fetchSuggestions(value)
  }

  const handleSuggestionClick = (suggestion: string) => {
      setQuery(suggestion)
      setSuggestions([])
      setShowSuggestions(false)
      // Trigger search immediately
      handleTextSearch(suggestion)
  }

  const handleTextSearch = async (overrideQuery?: string) => {
    const q = overrideQuery || query
    if (!q) return
    setLoading(true)
    setShowSuggestions(false)
    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
      console.log("SmartSearch: Using API URL:", API_URL)
      console.log("SmartSearch: Querying for:", q)
      
      const res = await axios.get(`${API_URL}/search?q=${q}&limit=20`)
      setResults(res.data.results)
    } catch (error) {
      console.error("Search failed", error)
    } finally {
      setLoading(false)
    }
  }

  const openProductDetail = (product: any) => {
      setDetailProduct(product)
      setIsDetailOpen(true)
  }

  const handleImageSearch = async () => {
    if (!imageFile) return
    setLoading(true)
    const formData = new FormData()
    formData.append("file", imageFile)
    
    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
      const res = await axios.post(`${API_URL}/search/image?limit=20`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      })
      setResults(res.data.results)
      setHasSearched(true)
    } catch (error) {
      console.error("Image search failed", error)
    } finally {
      setLoading(false)
    }
  }

  const handleRecommend = async (product: any) => {
    setSelectedProduct(product)
    setIsDialogOpen(true)
    setRecLoading(true)
    try {
        const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
        const res = await axios.get(`${API_URL}/recommend/${product.id}`)
        setRecommendations(res.data.recommendations)
        setMatchingItems(res.data.matching_items || [])
    } catch (error) {
        console.error("Recommendation failed", error)
    } finally {
        setRecLoading(false)
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Smart Fashion Search</CardTitle>
        <CardDescription>Search by text or upload an image to find similar items.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2 relative">
          <div className="relative w-full">
              <Input 
                placeholder="Search for 'red dress'..." 
                value={query} 
                onChange={handleInputChange}
                onKeyDown={(e) => e.key === "Enter" && handleTextSearch()}
                onFocus={() => query.length >= 2 && setShowSuggestions(true)}
                onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
              />
              {showSuggestions && suggestions.length > 0 && (
                  <ul className="absolute z-10 w-full bg-popover border rounded-md shadow-lg mt-1 max-h-60 overflow-auto">
                      {suggestions.map((s, idx) => (
                          <li 
                            key={idx} 
                            className="px-4 py-2 hover:bg-accent hover:text-accent-foreground cursor-pointer text-sm"
                            onClick={() => handleSuggestionClick(s)}
                          >
                              {s}
                          </li>
                      ))}
                  </ul>
              )}
          </div>
          <Button onClick={() => handleTextSearch()} disabled={loading}>
            {loading ? <Loader2 className="animate-spin" /> : <Search className="w-4 h-4" />}
          </Button>
        </div>
        
        <div className="flex items-center gap-2">
          <Input 
            type="file" 
            accept="image/*" 
            onChange={(e) => setImageFile(e.target.files?.[0] || null)} 
          />
          <Button variant="outline" onClick={handleImageSearch} disabled={!imageFile || loading}>
            <Upload className="w-4 h-4 mr-2" /> Image Search
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4">
          {results.map((item, idx) => (
            <Card key={idx} className="overflow-hidden flex flex-col group cursor-pointer hover:shadow-lg transition-shadow" onClick={() => openProductDetail(item)}>
              <div className="h-48 bg-muted overflow-hidden relative">
                {item.image_url ? (
                   <img 
                    src={item.image_url} 
                    alt={item.name} 
                    className="w-full h-full object-cover transition-transform group-hover:scale-105" 
                   />
                ) : (
                   <div className="w-full h-full flex items-center justify-center text-muted-foreground">Product Image</div>
                )}
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors" />
              </div>
              <CardContent className="p-4 flex-1 flex flex-col">
                <h3 className="font-bold">{item.name}</h3>
                <p className="text-sm text-muted-foreground mb-2 line-clamp-2">{item.description}</p>
                <div className="mt-auto flex justify-between items-center">
                    <p className="text-xs font-semibold text-primary">{item.category}</p>
                    <Button size="sm" variant="ghost" className="text-xs z-10 hover:bg-primary/10" onClick={(e) => {
                        e.stopPropagation()
                        e.preventDefault()
                        handleRecommend(item)
                    }}>
                        <Sparkles className="w-3 h-3 mr-1" /> Similar
                    </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Product Detail Dialog */}
        <Dialog open={isDetailOpen} onOpenChange={setIsDetailOpen}>
            <DialogContent className="max-w-4xl h-[80vh] flex flex-col md:flex-row gap-6 overflow-hidden z-[100]">
                {detailProduct && (
                    <>
                        <div className="w-full md:w-1/2 bg-muted rounded-lg overflow-hidden flex items-center justify-center relative">
                             {detailProduct.image_url ? (
                                <img
                                    src={detailProduct.image_url}
                                    alt={detailProduct.name}
                                    className="w-full h-full object-contain"
                                    style={{ imageRendering: "pixelated" }}
                                />
                             ) : (
                                <div className="text-muted-foreground">No Image</div>
                             )}
                        </div>
                        <div className="w-full md:w-1/2 flex flex-col overflow-y-auto">
                            <DialogHeader>
                                <DialogTitle className="text-2xl font-bold">{detailProduct.name}</DialogTitle>
                                <DialogDescription className="text-lg text-primary font-medium mt-2">
                                    {detailProduct.category}
                                </DialogDescription>
                            </DialogHeader>
                            
                            <div className="mt-6 space-y-4">
                                <div>
                                    <h4 className="font-semibold mb-2">Description</h4>
                                    <p className="text-muted-foreground leading-relaxed">
                                        {detailProduct.description}
                                    </p>
                                </div>
                                
                                <div className="pt-6 mt-auto">
                                    <Button className="w-full mb-2" onClick={() => {
                                        setIsDetailOpen(false)
                                        handleRecommend(detailProduct)
                                    }}>
                                        <Sparkles className="mr-2 h-4 w-4" /> Find Similar Items
                                    </Button>
                                    <Button variant="outline" className="w-full" onClick={() => setIsDetailOpen(false)}>
                                        Close
                                    </Button>
                                </div>
                            </div>
                        </div>
                    </>
                )}
            </DialogContent>
        </Dialog>

        {/* Recommendations Dialog */}
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
            <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto flex flex-col z-[100]">
                <DialogHeader>
                    <DialogTitle>Recommended for you</DialogTitle>
                    <DialogDescription>
                        Because you liked <span className="font-semibold">{selectedProduct?.name}</span>
                    </DialogDescription>
                </DialogHeader>
                
                {recLoading ? (
                    <div className="flex justify-center p-8">
                        <Loader2 className="animate-spin w-8 h-8 text-muted-foreground" />
                    </div>
                ) : (
                    <div className="space-y-6">
                        {/* Similar Items Section */}
                        <div>
                            <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                                <Sparkles className="w-4 h-4 text-primary" /> Similar Items
                            </h3>
                            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                                {recommendations.map((item, idx) => (
                                    <Card key={idx} className="overflow-hidden flex flex-col cursor-pointer hover:shadow-md transition-shadow" onClick={() => {
                                        setIsDialogOpen(false)
                                        openProductDetail(item)
                                    }}>
                                        <div className="h-32 bg-muted/50 overflow-hidden relative">
                                            {item.image_url ? (
                                                <img
                                                    src={item.image_url}
                                                    alt={item.name}
                                                    className="w-full h-full object-cover"
                                                    style={{ imageRendering: "pixelated" }}
                                                />
                                            ) : (
                                                <div className="w-full h-full flex items-center justify-center text-muted-foreground/50 text-sm">Product Image</div>
                                            )}
                                        </div>
                                        <CardContent className="p-3">
                                            <h4 className="font-bold text-sm line-clamp-1">{item.name}</h4>
                                            <p className="text-xs text-muted-foreground line-clamp-1">{item.description}</p>
                                        </CardContent>
                                    </Card>
                                ))}
                                {recommendations.length === 0 && (
                                    <p className="col-span-3 text-center text-muted-foreground text-sm py-4">No similar items found.</p>
                                )}
                            </div>
                        </div>

                        {/* Complete the Look Section */}
                        {matchingItems.length > 0 && (
                            <div>
                                <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                                    <Sparkles className="w-4 h-4 text-purple-500" /> Complete the Look
                                </h3>
                                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                                    {matchingItems.map((item, idx) => (
                                        <Card key={`match-${idx}`} className="overflow-hidden flex flex-col cursor-pointer hover:shadow-md transition-shadow border-purple-200" onClick={() => {
                                            setIsDialogOpen(false)
                                            openProductDetail(item)
                                        }}>
                                            <div className="h-32 bg-muted/50 overflow-hidden relative">
                                                {item.image_url ? (
                                                    <img
                                                        src={item.image_url}
                                                        alt={item.name}
                                                        className="w-full h-full object-cover"
                                                        style={{ imageRendering: "pixelated" }}
                                                    />
                                                ) : (
                                                    <div className="w-full h-full flex items-center justify-center text-muted-foreground/50 text-sm">Product Image</div>
                                                )}
                                                <div className="absolute top-1 right-1 bg-purple-500 text-white text-[10px] px-1.5 py-0.5 rounded-full font-bold">
                                                    Match
                                                </div>
                                            </div>
                                            <CardContent className="p-3">
                                                <h4 className="font-bold text-sm line-clamp-1">{item.name}</h4>
                                                <p className="text-xs text-muted-foreground line-clamp-1">{item.description}</p>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                            </div>
                        )}
                        
                        <div className="flex justify-end pt-2">
                             <Button variant="outline" onClick={() => setIsDialogOpen(false)}>Close</Button>
                        </div>
                    </div>
                )}
            </DialogContent>
        </Dialog>

      </CardContent>
    </Card>
  )
}

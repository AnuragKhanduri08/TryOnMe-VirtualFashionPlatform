"use client"

import { useState, useEffect } from "react"
import axios from "axios"
import SmartSearch from "@/components/SmartSearch"
import BodyMeasurement from "@/components/BodyMeasurement"
import MonitoringDashboard from "@/components/MonitoringDashboard"
import VirtualTryOn from "@/components/VirtualTryOn"
import ProductCard from "@/components/ProductCard"
import {
    NavigationMenu,
    NavigationMenuContent,
    NavigationMenuItem,
    NavigationMenuLink,
    NavigationMenuList,
    NavigationMenuTrigger,
    navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu"
import { ModeToggle } from "@/components/theme-toggle"
import { Dialog, DialogContent, DialogTitle, DialogDescription, DialogHeader } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Search, ShoppingBag, Shirt, Loader2, Sparkles, Camera, Ruler, LayoutDashboard, ScanSearch, Menu, X } from "lucide-react"
import { Input } from "@/components/ui/input"

interface Product {
    id: string | number
    name: string
    category: string
    subCategory?: string
    masterCategory?: string
    image_url: string
    price?: number
    gender?: string
}

export default function Home() {
  const [activeTab, setActiveTab] = useState("Men")
  const [products, setProducts] = useState<Product[]>([])
  const [loading, setLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string>("") // Debug State
  
  const [selectedCategory, setSelectedCategory] = useState("All")
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<Product[]>([])
  const [showDropdown, setShowDropdown] = useState(false)
  const [showHero, setShowHero] = useState(true)
  
  // View State: 'shop' | 'tryon' | 'search' | 'measure' | 'dashboard'
  const [currentView, setCurrentView] = useState("shop")

  // Try-On Modal State (for product-specific try-on)
  // We keep this to handle "Try On" button clicks from Product Cards
  const [tryOnProduct, setTryOnProduct] = useState<Product | undefined>(undefined)

  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null)
  const [detailOpen, setDetailOpen] = useState(false)
  
  // Recommendation State
  const [recOpen, setRecOpen] = useState(false)
  const [recommendations, setRecommendations] = useState<Product[]>([])
  const [matchingItems, setMatchingItems] = useState<Product[]>([])
  const [recLoading, setRecLoading] = useState(false)
  
  // Mobile Menu State
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  // Carousel State
  const [currentSlide, setCurrentSlide] = useState(0)
  const slides = [
      {
          image: "https://images.unsplash.com/photo-1523381210434-271e8be1f52b?q=80&w=2070&auto=format&fit=crop",
          title: "TryOnMe Exclusive Collection",
          subtitle: "Experience the future of digital fashion. Curated styles, virtual try-on.",
          cta: "Explore Collection",
          action: () => setSelectedCategory("Topwear"),
          align: "center"
      },
      {
          image: "https://images.unsplash.com/photo-1483985988355-763728e1935b?q=80&w=2070&auto=format&fit=crop",
          title: "Experience Virtual Try-On",
          subtitle: "Don't guess the fit. See exactly how it looks on you before you buy.",
          cta: "Try it Now",
          action: () => setCurrentView("tryon"),
          align: "left",
          badge: "New Feature"
      }
  ]

  // Auto-slide effect
  useEffect(() => {
      const timer = setInterval(() => {
          setCurrentSlide((prev) => (prev + 1) % slides.length)
      }, 5000)
      return () => clearInterval(timer)
  }, [slides.length])

  // Categories based on dataset
  const subCategories = ["Topwear", "Bottomwear", "Innerwear", "Shoes", "Watches", "Accessories"]
  const categories = ["All", ...subCategories]
  const brands = [
      { name: "Nike", image: "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Logo_NIKE.svg/1200px-Logo_NIKE.svg.png" },
      { name: "Puma", image: "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Puma-logo-%28text%29.svg/1200px-Puma-logo-%28text%29.svg.png" },
      { name: "Levi's", image: "https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Levi%27s_logo.svg/2560px-Levi%27s_logo.svg.png" },
      { name: "Adidas", image: "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Adidas_Logo.svg/1200px-Adidas_Logo.svg.png" },
      { name: "United Colors of Benetton", image: "/benetton.png" },
      { name: "Fabindia", image: "/fabindia.jpg" },
  ]

  useEffect(() => {
    fetchProducts()
  }, [activeTab, selectedCategory])

  const fetchProducts = async (query = "") => {
    setLoading(true)
    try {
      const rawUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
      const API_URL = rawUrl.trim()
      console.log("HomePage: Using API URL:", API_URL)
      
      let url = ""
      if (query) {
          url = `${API_URL}/search?q=${query}&limit=50`
      } else {
          // Increase limit to 100 to support home page sections
          url = `${API_URL}/products?limit=100&gender=${activeTab}`
          
          if (selectedCategory !== "All") {
            if (selectedCategory === "Shoes") {
                // Map "Shoes" UI category to "Footwear" masterCategory to include Shoes, Sandals, Flip Flops
                url += `&masterCategory=Footwear`
            } else if (selectedCategory === "Accessories") {
                // Map "Accessories" UI category to "Accessories" masterCategory
                url += `&masterCategory=Accessories`
            } else if (selectedCategory === "Watches") {
                 // "Watches" is a subCategory (and also a category), subCategory is safer
                 url += `&subCategory=Watches`
            } else {
                // "Topwear", "Bottomwear" map directly to subCategory
                url += `&subCategory=${selectedCategory}`
            }
          }
      }
      
      console.log("HomePage: Fetching URL:", url)
      const res = await axios.get(url)
      
      console.log("HomePage: Full API Response:", res.data) // DEBUG LOG
      
      // Search endpoint returns { results: [...] }, products endpoint returns [...]
      const data = res.data.results || res.data
      
      if (Array.isArray(data)) {
          console.log(`HomePage: Parsed ${data.length} products`)
          setProducts(data)
      } else {
          console.error("HomePage: API returned non-array data", data)
          setProducts([])
      }
      
      setErrorMsg("") // Clear error on success
    } catch (error: any) {
      console.error("Failed to fetch products", error)
      setProducts([])
      setErrorMsg(error.message || "Unknown Fetch Error")
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = async (e: React.ChangeEvent<HTMLInputElement>) => {
      const query = e.target.value
      setSearchQuery(query)
      if (query.length > 0) setShowHero(false)
      
      if (query.length > 1) {
          try {
              const rawUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
              const API_URL = rawUrl.trim()
              const res = await axios.get(`${API_URL}/search?q=${query}&limit=5`)
              const data = res.data.results || res.data
              setSearchResults(Array.isArray(data) ? data : [])
              setShowDropdown(true)
          } catch (error) {
              console.error("Search suggestion failed", error)
          }
      } else {
          setSearchResults([])
          setShowDropdown(false)
      }
  }

  const handleSearchSubmit = (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
          setShowDropdown(false)
          fetchProducts(searchQuery)
      }
  }

  const handleSuggestionClick = (product: Product) => {
      setSearchQuery(product.name)
      setShowDropdown(false)
      setShowHero(false)
      fetchProducts(product.name)
  }

  const handleBrandClick = (brandName: string) => {
    setSearchQuery(brandName)
    setShowHero(false)
    fetchProducts(brandName)
    // Scroll to products
    window.scrollTo({ top: 500, behavior: 'smooth' })
  }

  const handleTryOn = (product: Product) => {
    setTryOnProduct(product)
    setCurrentView("tryon")
  }

  const handleProductClick = (product: Product) => {
      setSelectedProduct(product)
      setDetailOpen(true)
  }

  const handleSimilar = async (product: Product) => {
      setSelectedProduct(product)
      setRecOpen(true)
      setRecLoading(true)
      try {
          const rawUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
          const API_URL = rawUrl.trim()
          console.log(`Getting recommendations for product ${product.id} from ${API_URL}...`)
          
          const res = await axios.get(`${API_URL}/recommend/${product.id}`)
          console.log("Recommendation Response:", res.data)
          
          setRecommendations(res.data.recommendations || [])
          setMatchingItems(res.data.matching_items || [])
      } catch (error) {
          console.error("Recommendation failed", error)
          setRecommendations([])
          setMatchingItems([])
      } finally {
          setRecLoading(false)
      }
  }

  return (
    <main className="relative min-h-screen bg-background text-foreground transition-colors duration-300">
      {/* DEBUG BANNER - Remove before final production */}
      <div className="bg-yellow-100 border-b border-yellow-200 p-2 text-xs text-black font-mono overflow-auto">
        <p><strong>DEBUG INFO:</strong></p>
        <p>API URL Env: {process.env.NEXT_PUBLIC_API_URL ? `"${process.env.NEXT_PUBLIC_API_URL}"` : "undefined"}</p>
        <p>Fallback URL: {(process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").trim()}</p>
        {errorMsg && <p className="text-red-600 font-bold">Last Error: {errorMsg}</p>}
        <p>Current View: {currentView} | Loading: {loading.toString()} | Products: {products.length}</p>
      </div>

      {/* Navbar */}
      <nav className="sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between px-4">
            <div className="flex items-center gap-6">
                <a 
                    href="#" 
                    onClick={(e) => {
                        e.preventDefault()
                        setCurrentView("shop")
                        setSelectedCategory("All")
                        setSearchQuery("")
                        setActiveTab("Men") // Reset to default gender or keep current? Usually home resets to default.
                        setShowHero(true)
                        fetchProducts("") 
                    }} 
                    className="flex items-center gap-2 font-bold text-xl tracking-tight cursor-pointer"
                >
                    <img src="/favicon.ico" alt="Logo" className="w-8 h-8 rounded-lg" />
                    TryOnMe
                </a>
                
                <div className="hidden md:flex gap-1">
                    <NavigationMenu>
                        <NavigationMenuList>
                            {["Men", "Women", "Boys", "Girls"].map((gender) => (
                                <NavigationMenuItem key={gender}>
                                    <NavigationMenuTrigger className="bg-transparent">
                                        {gender}
                                    </NavigationMenuTrigger>
                                    <NavigationMenuContent>
                                        <ul className="grid w-[400px] gap-3 p-4 md:w-[500px] md:grid-cols-2 lg:w-[600px]">
                                            {subCategories.map((sub) => (
                                                <li key={sub}>
                                                    <NavigationMenuLink asChild>
                                                        <a
                                                            className="block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground cursor-pointer"
                                                            onClick={() => {
                                                                setActiveTab(gender)
                                                                setSelectedCategory(sub)
                                                                setCurrentView("shop")
                                                                setShowHero(false)
                                                            }}
                                                        >
                                                            <div className="text-sm font-medium leading-none">{sub}</div>
                                                            <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
                                                                Shop {gender}'s {sub} collection
                                                            </p>
                                                        </a>
                                                    </NavigationMenuLink>
                                                </li>
                                            ))}
                                        </ul>
                                    </NavigationMenuContent>
                                </NavigationMenuItem>
                            ))}
                        </NavigationMenuList>
                    </NavigationMenu>
                </div>
            </div>

            <div className="flex items-center gap-4">
                {/* Mobile Menu Toggle */}
                <div className="md:hidden">
                    <Button variant="ghost" size="icon" onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
                        {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
                    </Button>
                </div>
                <div className="hidden lg:flex items-center gap-1 mr-2">
                    <Button 
                        variant={currentView === "tryon" ? "secondary" : "ghost"} 
                        size="sm" 
                        onClick={() => { setTryOnProduct(undefined); setCurrentView("tryon"); }} 
                        className="gap-2 font-medium"
                    >
                        <Camera className="w-4 h-4" />
                        <span className="hidden xl:inline">Virtual Try-On</span>
                    </Button>
                    <Button 
                        variant={currentView === "search" ? "secondary" : "ghost"} 
                        size="sm" 
                        onClick={() => setCurrentView("search")} 
                        className="gap-2 font-medium"
                    >
                        <ScanSearch className="w-4 h-4" />
                        <span className="hidden xl:inline">AI Search</span>
                    </Button>
                    <Button 
                        variant={currentView === "measure" ? "secondary" : "ghost"} 
                        size="sm" 
                        onClick={() => setCurrentView("measure")} 
                        className="gap-2 font-medium"
                    >
                        <Ruler className="w-4 h-4" />
                        <span className="hidden xl:inline">Size Guide</span>
                    </Button>
                    <Button 
                        variant={currentView === "dashboard" ? "secondary" : "ghost"} 
                        size="sm" 
                        onClick={() => setCurrentView("dashboard")} 
                        className="gap-2 font-medium"
                    >
                        <LayoutDashboard className="w-4 h-4" />
                        <span className="hidden xl:inline">Dashboard</span>
                    </Button>
                </div>

                <div className="relative hidden sm:block w-64 group">
                    <div className="relative">
                        <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground z-10" />
                        <Input 
                            placeholder="Search for items..." 
                            className="pl-9 h-9" 
                            value={searchQuery}
                            onChange={handleSearch}
                            onKeyDown={handleSearchSubmit}
                            onFocus={() => searchQuery.length > 1 && setShowDropdown(true)}
                            onBlur={() => setTimeout(() => setShowDropdown(false), 200)}
                        />
                    </div>
                    
                    {/* Search Dropdown */}
                    {showDropdown && searchResults.length > 0 && (
                        <div className="absolute top-full left-0 right-0 mt-2 bg-popover text-popover-foreground rounded-md border shadow-md overflow-hidden z-50">
                            <div className="py-1">
                                {searchResults.map((item) => (
                                    <button
                                        key={item.id}
                                        className="w-full text-left px-4 py-2 text-sm hover:bg-muted flex items-center gap-3 transition-colors"
                                        onClick={() => handleSuggestionClick(item)}
                                    >
                                        <div className="relative w-8 h-8 rounded overflow-hidden bg-muted">
                                            <img src={item.image_url} alt="" className="object-cover w-full h-full" />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <p className="font-medium truncate">{item.name}</p>
                                            <p className="text-xs text-muted-foreground truncate">{item.category} • {item.gender}</p>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
                <ModeToggle />
            </div>
        </div>
      </nav>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden fixed top-16 left-0 right-0 bg-background border-b z-30 p-4 shadow-lg animate-in slide-in-from-top-5">
            <div className="flex flex-col gap-4">
                <div className="relative">
                     <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground z-10" />
                     <Input 
                        placeholder="Search..." 
                        className="pl-9 h-10 w-full" 
                        value={searchQuery}
                        onChange={handleSearch}
                        onKeyDown={handleSearchSubmit}
                     />
                </div>
                
                <div className="grid grid-cols-2 gap-2">
                    <Button variant="outline" onClick={() => { setCurrentView("tryon"); setMobileMenuOpen(false); }} className="justify-start">
                        <Camera className="mr-2 h-4 w-4" /> Try-On
                    </Button>
                    <Button variant="outline" onClick={() => { setCurrentView("search"); setMobileMenuOpen(false); }} className="justify-start">
                        <ScanSearch className="mr-2 h-4 w-4" /> AI Search
                    </Button>
                    <Button variant="outline" onClick={() => { setCurrentView("measure"); setMobileMenuOpen(false); }} className="justify-start">
                        <Ruler className="mr-2 h-4 w-4" /> Size Guide
                    </Button>
                    <Button variant="outline" onClick={() => { setCurrentView("dashboard"); setMobileMenuOpen(false); }} className="justify-start">
                        <LayoutDashboard className="mr-2 h-4 w-4" /> Dashboard
                    </Button>
                </div>

                <div className="border-t pt-4">
                    <h3 className="font-semibold mb-2 text-sm text-muted-foreground">Categories</h3>
                    <div className="flex flex-wrap gap-2">
                        {["Men", "Women", "Boys", "Girls"].map((gender) => (
                            <Button 
                                key={gender} 
                                variant={activeTab === gender ? "default" : "ghost"} 
                                size="sm"
                                onClick={() => {
                                    setActiveTab(gender)
                                    setMobileMenuOpen(false)
                                    setCurrentView("shop")
                                    setShowHero(false)
                                    fetchProducts("")
                                }}
                            >
                                {gender}
                            </Button>
                        ))}
                    </div>
                </div>
            </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 w-full flex flex-col min-h-[calc(100vh-64px)]">
        {currentView === "shop" && (
            <div className="container px-4 py-8">
                {/* Hero / Brand Showcase (Only when All categories selected and no search) */}
                {selectedCategory === "All" && searchQuery === "" && showHero && (
                    <div className="mb-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
                        {/* Carousel Hero */}
                        <div className="relative w-full h-[400px] rounded-2xl overflow-hidden mb-8 group">
                            {slides.map((slide, index) => (
                                <div 
                                    key={index}
                                    className={`absolute inset-0 transition-opacity duration-1000 ${index === currentSlide ? 'opacity-100 z-10' : 'opacity-0 z-0'}`}
                                >
                                    <img 
                                        src={slide.image} 
                                        alt={slide.title} 
                                        className="w-full h-full object-cover brightness-50"
                                    />
                                    <div className={`absolute inset-0 flex flex-col justify-center text-white p-8 md:p-16 ${slide.align === 'center' ? 'items-center text-center' : 'items-start text-left'}`}>
                                        {slide.badge && (
                                            <div className="bg-primary/20 backdrop-blur-sm px-3 py-1 rounded-full text-xs font-bold mb-4 uppercase tracking-wider border border-primary/50 animate-in fade-in slide-in-from-bottom-2">
                                                {slide.badge}
                                            </div>
                                        )}
                                        <h1 className="text-4xl md:text-6xl font-bold mb-4 tracking-tight max-w-3xl animate-in fade-in slide-in-from-bottom-3">{slide.title}</h1>
                                        <p className="text-lg md:text-xl mb-8 max-w-2xl opacity-90 animate-in fade-in slide-in-from-bottom-4">{slide.subtitle}</p>
                                        <Button size="lg" className="bg-white text-black hover:bg-white/90 gap-2 animate-in fade-in slide-in-from-bottom-5" onClick={slide.action}>
                                            {slide.cta} {slide.badge && <Camera className="w-4 h-4" />}
                                        </Button>
                                    </div>
                                </div>
                            ))}
                            
                            {/* Dots Indicator */}
                            <div className="absolute bottom-6 left-0 right-0 z-20 flex justify-center gap-2">
                                {slides.map((_, index) => (
                                    <button
                                        key={index}
                                        onClick={() => setCurrentSlide(index)}
                                        className={`w-2.5 h-2.5 rounded-full transition-all ${index === currentSlide ? 'bg-white w-8' : 'bg-white/50 hover:bg-white/80'}`}
                                        aria-label={`Go to slide ${index + 1}`}
                                    />
                                ))}
                            </div>
                        </div>

                        <h2 className="text-2xl font-bold mb-6 tracking-tight">Featured Brands</h2>
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                            {brands.map((brand, idx) => (
                                <div key={idx} className="group relative aspect-square rounded-xl overflow-hidden cursor-pointer border bg-white hover:shadow-lg transition-all duration-300" onClick={() => handleBrandClick(brand.name)}>
                                    <div className="absolute inset-0 p-8 flex items-center justify-center">
                                        <img 
                                            src={brand.image} 
                                            alt={brand.name} 
                                            className="w-full h-full object-contain transition-transform duration-500 group-hover:scale-110"
                                            onError={(e) => {
                                                // Fallback to text if image fails
                                                const target = e.currentTarget
                                                target.style.display = 'none'
                                                if (target.parentElement) {
                                                    target.parentElement.innerText = brand.name
                                                    target.parentElement.className = "absolute inset-0 p-4 flex items-center justify-center text-center font-bold text-sm text-gray-800"
                                                }
                                            }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Shop by Category (Roundels) - Only on Home View */}
                {selectedCategory === "All" && searchQuery === "" && (
                    <div className="mb-12">
                        <h2 className="text-2xl font-bold mb-6 tracking-tight">Shop by Category</h2>
                        <div className="flex gap-6 overflow-x-auto pb-4 scrollbar-hide">
                            {[
                                { name: "Topwear", image: "https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=500&auto=format&fit=crop&q=60" },
                                { name: "Bottomwear", image: "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=500&auto=format&fit=crop&q=60" },
                                { name: "Shoes", image: "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=500&auto=format&fit=crop&q=60" },
                                { name: "Watches", image: "https://images.unsplash.com/photo-1524592094714-0f0654e20314?w=500&auto=format&fit=crop&q=60" },
                                { name: "Accessories", image: "https://images.unsplash.com/photo-1576053139778-7e32f2ae3cfd?w=500&auto=format&fit=crop&q=60" },
                                { name: "Innerwear", image: "/innerwear.jpg" },
                            ].map((cat) => (
                                <div 
                                    key={cat.name} 
                                    className="flex flex-col items-center gap-3 min-w-[100px] cursor-pointer group"
                                    onClick={() => {
                                        setSelectedCategory(cat.name)
                                        setShowHero(false)
                                    }}
                                >
                                    <div className="w-24 h-24 rounded-full overflow-hidden border-2 border-transparent group-hover:border-primary transition-all p-0.5">
                                        <div className="w-full h-full rounded-full overflow-hidden">
                                            <img 
                                                src={cat.image} 
                                                alt={cat.name} 
                                                className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                                            />
                                        </div>
                                    </div>
                                    <span className="font-medium text-sm">{cat.name}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Category Filters (Hidden if in Home mode) */}
                {/* We keep filters visible but below Hero if active, or just hide hero when specific cat selected */}
                
                {selectedCategory !== "All" && (
                    <div className="flex flex-col gap-4 mb-8">
                         <div className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
                            <span className="cursor-pointer hover:text-foreground" onClick={() => {
                                setSelectedCategory("All")
                                setShowHero(true)
                            }}>Home</span>
                            <span>/</span>
                            <span className="font-medium text-foreground">{selectedCategory}</span>
                        </div>
                        
                        <div className="flex items-center justify-between">
                            <h2 className="text-3xl font-bold tracking-tight">Shop {selectedCategory}</h2>
                            <Button variant="outline" onClick={() => {
                                setSelectedCategory("All")
                                setShowHero(true)
                            }}>
                                Back to Home
                            </Button>
                        </div>
                        
                        <div className="flex items-center gap-2 overflow-x-auto pb-2 scrollbar-hide">
                        {categories.map((cat) => (
                            <Button
                                key={cat}
                                variant={selectedCategory === cat ? "default" : "outline"}
                                size="sm"
                                onClick={() => {
                                    setSelectedCategory(cat)
                                    setShowHero(false)
                                }}
                                className="whitespace-nowrap rounded-full px-6"
                            >
                                {cat}
                            </Button>
                        ))}
                        </div>
                    </div>
                )}

                {/* Home Page Sections (Horizontal Rails) */}
                {selectedCategory === "All" && searchQuery === "" ? (
                    <div className="space-y-12 relative min-h-[50vh]">
                        {loading && (
                            <div className="absolute inset-0 z-10 bg-background/50 backdrop-blur-[1px] flex items-center justify-center">
                                <Loader2 className="h-10 w-10 animate-spin text-primary" />
                            </div>
                        )}

                        {/* Section 1: Trending Topwear */}
                        <div>
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-xl font-bold">Trending in Topwear</h3>
                                <Button variant="link" className="text-primary" onClick={() => setSelectedCategory("Topwear")}>View All</Button>
                            </div>
                            <div className="flex gap-4 overflow-x-auto pb-4 scrollbar-hide snap-x">
                                {products
                                    .filter(p => p.subCategory === "Topwear")
                                    .slice(0, 10)
                                    .map((product) => (
                                        <div key={product.id} className="min-w-[200px] w-[200px] md:min-w-[240px] md:w-[240px] snap-start h-full">
                                            <ProductCard 
                                                product={product} 
                                                onTryOn={handleTryOn} 
                                                onClick={handleProductClick}
                                                onSimilar={handleSimilar}
                                            />
                                        </div>
                                    ))}
                            </div>
                        </div>

                        {/* Section 2: Best in Bottomwear */}
                        <div>
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-xl font-bold">Best in Bottomwear</h3>
                                <Button variant="link" className="text-primary" onClick={() => setSelectedCategory("Bottomwear")}>View All</Button>
                            </div>
                            <div className="flex gap-4 overflow-x-auto pb-4 scrollbar-hide snap-x">
                                {products
                                    .filter(p => p.subCategory === "Bottomwear")
                                    .slice(0, 10)
                                    .map((product) => (
                                        <div key={product.id} className="min-w-[200px] w-[200px] md:min-w-[240px] md:w-[240px] snap-start h-full">
                                            <ProductCard 
                                                product={product} 
                                                onTryOn={handleTryOn} 
                                                onClick={handleProductClick}
                                                onSimilar={handleSimilar}
                                            />
                                        </div>
                                    ))}
                            </div>
                        </div>

                        {/* Section 3: Footwear Favorites */}
                        <div>
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-xl font-bold">Footwear Favorites</h3>
                                <Button variant="link" className="text-primary" onClick={() => setSelectedCategory("Shoes")}>View All</Button>
                            </div>
                            <div className="flex gap-4 overflow-x-auto pb-4 scrollbar-hide snap-x">
                                {products
                                    .filter(p => p.subCategory === "Shoes" || p.masterCategory === "Footwear")
                                    .slice(0, 10)
                                    .map((product) => (
                                        <div key={product.id} className="min-w-[200px] w-[200px] md:min-w-[240px] md:w-[240px] snap-start h-full">
                                            <ProductCard 
                                                product={product} 
                                                onTryOn={handleTryOn} 
                                                onClick={handleProductClick}
                                                onSimilar={handleSimilar}
                                            />
                                        </div>
                                    ))}
                            </div>
                        </div>
                    </div>
                ) : (
                    /* Existing Grid View (for specific categories/search) */
                    loading ? (
                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6">
                        {[...Array(10)].map((_, i) => (
                            <div key={i} className="aspect-[3/4] bg-muted/30 animate-pulse rounded-xl" />
                        ))}
                    </div>
                ) : (
                    <div className="min-h-[50vh]">
                        {products.length > 0 ? (
                            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6">
                                {products.map((product) => (
                                    <ProductCard 
                                        key={product.id} 
                                        product={product} 
                                        onTryOn={handleTryOn} 
                                        onClick={handleProductClick}
                                        onSimilar={handleSimilar}
                                    />
                                ))}
                            </div>
                        ) : (
                            <div className="text-center py-20 text-muted-foreground">
                                <p className="text-lg">No products found for this category.</p>
                                <Button variant="link" onClick={() => setSelectedCategory("All")}>Clear Filters</Button>
                            </div>
                        )}
                    </div>
                )
                )}
            </div>
        )}

        {currentView === "tryon" && (
            <div className="flex-1 relative min-h-[calc(100vh-64px)]">
                 <div 
                    className="absolute inset-0 z-0" 
                    style={{
                        backgroundImage: 'url("https://images.unsplash.com/photo-1558769132-cb1aea458c5e?q=80&w=2070&auto=format&fit=crop")',
                        backgroundSize: 'cover',
                        backgroundPosition: 'center',
                    }}
                 />
                 <div className="absolute inset-0 bg-background/50 z-0 backdrop-blur-[2px]" />
                 
                 <div className="container relative z-10 h-full max-w-6xl mx-auto flex flex-col py-8">
                    <div className="mb-6 flex items-center justify-between">
                         <div>
                            <h2 className="text-3xl font-bold tracking-tight">Virtual Try-On Room</h2>
                            <p className="text-muted-foreground mt-1">Experience your perfect fit in our digital studio.</p>
                         </div>
                         <Button variant="outline" onClick={() => setCurrentView("shop")} className="bg-background/50 backdrop-blur-sm hover:bg-background">Back to Shop</Button>
                    </div>
                    <div className="flex-1 border rounded-xl overflow-hidden bg-background/95 shadow-xl backdrop-blur-sm">
                        <VirtualTryOn initialProduct={tryOnProduct} />
                    </div>
                 </div>
            </div>
        )}

        {currentView === "search" && (
            <div className="flex-1 relative min-h-[calc(100vh-64px)]">
                 <div 
                    className="absolute inset-0 z-0" 
                    style={{
                        backgroundImage: 'url("https://images.unsplash.com/photo-1490481651871-ab68de25d43d?q=80&w=2070&auto=format&fit=crop")',
                        backgroundSize: 'cover',
                        backgroundPosition: 'center',
                    }}
                 />
                 <div className="absolute inset-0 bg-background/50 z-0 backdrop-blur-[2px]" />

                 <div className="container relative z-10 h-full max-w-5xl mx-auto flex flex-col py-8">
                    <div className="mb-6 flex items-center justify-between">
                         <div>
                            <h2 className="text-3xl font-bold tracking-tight">AI Smart Search</h2>
                            <p className="text-muted-foreground mt-1">Find exactly what you're looking for with AI.</p>
                         </div>
                         <Button variant="outline" onClick={() => setCurrentView("shop")} className="bg-background/50 backdrop-blur-sm hover:bg-background">Back to Shop</Button>
                    </div>
                    <div className="flex-1 bg-background/80 rounded-xl p-6 shadow-xl backdrop-blur-sm border">
                        <SmartSearch />
                    </div>
                 </div>
            </div>
        )}

        {currentView === "measure" && (
            <div className="flex-1 relative min-h-[calc(100vh-64px)]">
                 <div 
                    className="absolute inset-0 z-0" 
                    style={{
                        backgroundImage: 'url("https://images.unsplash.com/photo-1554568218-0f1715e72254?q=80&w=2070&auto=format&fit=crop")',
                        backgroundSize: 'cover',
                        backgroundPosition: 'center',
                    }}
                 />
                 <div className="absolute inset-0 bg-background/50 z-0 backdrop-blur-[2px]" />

                 <div className="container relative z-10 h-full max-w-5xl mx-auto flex flex-col py-8">
                    <div className="mb-6 flex items-center justify-between">
                         <div>
                            <h2 className="text-3xl font-bold tracking-tight">AI Size Guide</h2>
                            <p className="text-muted-foreground mt-1">Get precise measurements for the perfect fit.</p>
                         </div>
                         <Button variant="outline" onClick={() => setCurrentView("shop")} className="bg-background/50 backdrop-blur-sm hover:bg-background">Back to Shop</Button>
                    </div>
                    <div className="flex-1 bg-background/80 rounded-xl p-6 shadow-xl backdrop-blur-sm border">
                        <BodyMeasurement />
                    </div>
                 </div>
            </div>
        )}

        {currentView === "dashboard" && (
            <div className="flex-1 p-6 bg-muted/10 h-full">
                 <div className="container h-full max-w-6xl mx-auto flex flex-col">
                    <div className="mb-6 flex items-center justify-between">
                         <h2 className="text-2xl font-bold tracking-tight">System Dashboard</h2>
                         <Button variant="outline" onClick={() => setCurrentView("shop")}>Back to Shop</Button>
                    </div>
                    <div className="flex-1">
                        <MonitoringDashboard />
                    </div>
                 </div>
            </div>
        )}
      </div>

      {/* Product Detail Modal */}
      <Dialog open={detailOpen} onOpenChange={setDetailOpen}>
        <DialogContent className="max-w-4xl h-[80vh] flex flex-col md:flex-row gap-6 overflow-hidden z-[100]">
            {selectedProduct && (
                <>
                    <div className="w-full md:w-1/2 bg-muted rounded-lg overflow-hidden flex items-center justify-center relative">
                         {selectedProduct.image_url ? (
                            <img
                                src={selectedProduct.image_url}
                                alt={selectedProduct.name}
                                className="w-full h-full object-contain"
                                style={{ imageRendering: "pixelated" }}
                            />
                         ) : (
                            <div className="text-muted-foreground">No Image</div>
                         )}
                    </div>
                    <div className="w-full md:w-1/2 flex flex-col overflow-y-auto">
                        <DialogHeader>
                            <DialogTitle className="text-2xl font-bold">{selectedProduct.name}</DialogTitle>
                            <DialogDescription className="text-lg text-primary font-medium mt-2">
                                {selectedProduct.category} • {selectedProduct.gender}
                            </DialogDescription>
                        </DialogHeader>
                        
                        <div className="mt-6 space-y-4">
                            <div>
                                <h4 className="font-semibold mb-2">Details</h4>
                                <p className="text-muted-foreground leading-relaxed">
                                    High-quality fashion item suitable for {selectedProduct.gender}. 
                                    Perfect for your {selectedProduct.category} collection.
                                </p>
                            </div>
                            
                            <div className="pt-6 mt-auto space-y-3">
                                <Button className="w-full" onClick={() => {
                                    setDetailOpen(false)
                                    handleSimilar(selectedProduct)
                                }}>
                                    <Sparkles className="mr-2 h-4 w-4" /> Find Similar Items
                                </Button>
                                
                                {["Topwear", "Tops"].includes(selectedProduct.category) && (
                                    <Button variant="secondary" className="w-full" onClick={() => {
                                        setDetailOpen(false)
                                        handleTryOn(selectedProduct)
                                    }}>
                                        <Shirt className="mr-2 h-4 w-4" /> Try On Now
                                    </Button>
                                )}

                                <Button variant="outline" className="w-full" onClick={() => setDetailOpen(false)}>
                                    Close
                                </Button>
                            </div>
                        </div>
                    </div>
                </>
            )}
        </DialogContent>
      </Dialog>

      {/* Recommendations Modal */}
      <Dialog open={recOpen} onOpenChange={setRecOpen}>
            <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto flex flex-col z-[100]">
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
                    <div className="space-y-8 py-4">
                        {/* Similar Items Section */}
                        <div>
                            <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                                <Sparkles className="w-5 h-5 text-primary" /> Similar Items
                            </h3>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                {recommendations.map((item, idx) => (
                                    <div key={idx} className="group relative border rounded-lg overflow-hidden cursor-pointer hover:shadow-md transition-all" onClick={() => {
                                        setRecOpen(false)
                                        handleProductClick(item)
                                    }}>
                                        <div className="aspect-[3/4] bg-muted relative">
                                            <img
                                                src={item.image_url}
                                                alt={item.name}
                                                className="w-full h-full object-cover"
                                                style={{ imageRendering: "pixelated" }}
                                            />
                                        </div>
                                        <div className="p-3">
                                            <h4 className="font-medium text-sm line-clamp-1">{item.name}</h4>
                                            <p className="text-xs text-muted-foreground">{item.category}</p>
                                        </div>
                                    </div>
                                ))}
                                {recommendations.length === 0 && (
                                    <p className="col-span-4 text-center text-muted-foreground py-8">No similar items found.</p>
                                )}
                            </div>
                        </div>

                        {/* Complete the Look Section */}
                        {matchingItems.length > 0 && (
                            <div>
                                <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                                    <Sparkles className="w-5 h-5 text-purple-500" /> Complete the Look
                                </h3>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    {matchingItems.map((item, idx) => (
                                        <div key={`match-${idx}`} className="group relative border border-purple-200 rounded-lg overflow-hidden cursor-pointer hover:shadow-md transition-all" onClick={() => {
                                            setRecOpen(false)
                                            handleProductClick(item)
                                        }}>
                                            <div className="aspect-[3/4] bg-muted relative">
                                                <img
                                                    src={item.image_url}
                                                    alt={item.name}
                                                    className="w-full h-full object-cover"
                                                    style={{ imageRendering: "pixelated" }}
                                                />
                                                <div className="absolute top-2 right-2 bg-purple-500 text-white text-[10px] px-2 py-0.5 rounded-full font-bold">
                                                    Match
                                                </div>
                                            </div>
                                            <div className="p-3">
                                                <h4 className="font-medium text-sm line-clamp-1">{item.name}</h4>
                                                <p className="text-xs text-muted-foreground">{item.category}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </DialogContent>
        </Dialog>
    </main>
  );
}

import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ShoppingBag, Sparkles } from "lucide-react"

interface Product {
    id: string | number
    name: string
    category: string
    subCategory?: string
    image_url: string
    price?: number
    gender?: string
}

interface ProductCardProps {
    product: Product
    onTryOn: (product: Product) => void
    onClick: (product: Product) => void
    onSimilar: (product: Product) => void
}

export default function ProductCard({ product, onTryOn, onClick, onSimilar }: ProductCardProps) {
    
    // Check if product is Topwear (suitable for Try-On)
    // We check subCategory first as it is the most reliable field in this dataset
    const isTopwear = (product.subCategory === "Topwear") || 
                      product.category === "Topwear" || 
                      product.category === "Tops" || 
                      (product.category && product.category.toLowerCase().includes("top")) ||
                      (product.subCategory && product.subCategory.toLowerCase().includes("top"));

    return (
        <div 
            className="group relative bg-card border rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-all duration-300 flex flex-col h-full cursor-pointer"
            onClick={() => onClick(product)}
        >
            {/* Image Container */}
            <div className="relative aspect-[3/4] overflow-hidden bg-muted/30">
                <Image 
                    src={product.image_url} 
                    alt={product.name} 
                    fill
                    className="object-contain mix-blend-multiply dark:mix-blend-normal group-hover:scale-105 transition-transform duration-500"
                    style={{ imageRendering: "pixelated" }}
                />
                
                {/* Overlay Actions */}
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col items-center justify-center gap-2 p-4">
                    {isTopwear ? (
                        <Button 
                            onClick={(e) => {
                                e.stopPropagation()
                                onTryOn(product)
                            }} 
                            variant="secondary" 
                            size="sm"
                            className="w-full font-medium transform translate-y-4 group-hover:translate-y-0 transition-transform duration-300 delay-75"
                        >
                            <Sparkles className="w-4 h-4 mr-2" />
                            Try On
                        </Button>
                    ) : (
                        <Badge variant="secondary" className="transform translate-y-4 group-hover:translate-y-0 transition-transform duration-300 cursor-not-allowed opacity-80">
                            Try-On: Topwear Only
                        </Badge>
                    )}
                    
                    <Button 
                        onClick={(e) => {
                            e.stopPropagation()
                            onSimilar(product)
                        }}
                        variant="outline" 
                        size="sm"
                        className="w-full font-medium transform translate-y-4 group-hover:translate-y-0 transition-transform duration-300 delay-100 bg-background/80 backdrop-blur-sm"
                    >
                        <ShoppingBag className="w-4 h-4 mr-2" />
                        Similar
                    </Button>
                </div>

                <div className="absolute top-2 left-2">
                     {product.gender && (
                        <Badge variant="secondary" className="text-[10px] uppercase tracking-wider bg-white/80 dark:bg-black/60 backdrop-blur-sm">
                            {product.gender}
                        </Badge>
                     )}
                </div>
            </div>

            {/* Content */}
            <div className="p-4 flex flex-col flex-1">
                <h3 className="font-medium text-sm line-clamp-2 min-h-[40px] mb-1" title={product.name}>
                    {product.name}
                </h3>
                <p className="text-xs text-muted-foreground mb-3">{product.category}</p>
            </div>
        </div>
    )
}
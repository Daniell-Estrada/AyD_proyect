import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "../../../shared/utils"

/**
 * Button variants using class-variance-authority
 */
const buttonVariants = cva(
  "inline-flex items-center justify-start gap-2 whitespace-nowrap rounded-none border-0 bg-transparent text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring/40 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "text-primary hover:text-primary/80",
        destructive: "text-destructive hover:text-destructive/80",
        outline: "text-foreground hover:text-foreground/80",
        secondary: "text-muted-foreground hover:text-foreground",
        ghost: "text-foreground/80 hover:text-foreground",
        link: "text-primary underline underline-offset-4 hover:text-primary/80",
      },
      size: {
        default: "py-1",
        sm: "text-xs",
        lg: "text-lg py-2",
        icon: "p-1",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
)
interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  },
)
Button.displayName = "Button"

export { Button }

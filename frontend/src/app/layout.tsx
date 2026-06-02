import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800", "900"],
});

export const metadata: Metadata = {
  title: "RetinaAI — Diabetic Retinopathy Detection Platform",
  description:
    "Research-grade deep learning platform for automated diabetic retinopathy detection and severity classification using 6 state-of-the-art neural network architectures with Grad-CAM explainability.",
  keywords: [
    "diabetic retinopathy",
    "deep learning",
    "medical imaging",
    "retinal screening",
    "AI healthcare",
    "Grad-CAM",
    "EfficientNet",
    "ResNet",
    "ViT",
  ],
  authors: [{ name: "RetinaAI Research" }],
  openGraph: {
    title: "RetinaAI — Diabetic Retinopathy Detection",
    description: "AI-powered retinal screening with multi-model comparison and explainability",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} font-sans antialiased flex flex-col min-h-screen`}>
        <Navbar />
        <main className="flex-grow">{children}</main>
        <Footer />
      </body>
    </html>
  );
}

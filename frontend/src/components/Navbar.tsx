"use client";

import Link from "next/link";
import { Activity, Menu, X, LogOut, LogIn, UserCircle } from "lucide-react";
import { useState, useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { getToken, removeToken, isTokenExpired, decodeToken } from "@/lib/auth";

const protectedLinks = [
  { href: "/", label: "Predict" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/training", label: "Training" },
  { href: "/research", label: "Research" },
  { href: "/dataset", label: "Dataset" },
  { href: "/how-it-works", label: "How It Works" },
  { href: "/history", label: "History" },
  { href: "/about", label: "About" },
];

const publicLinks = [
  { href: "/login", label: "Sign In" },
  { href: "/register", label: "Register" },
];

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [username, setUsername] = useState<string | null>(null);
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Check auth state on pathname change
  useEffect(() => {
    const token = getToken();
    if (token && !isTokenExpired(token)) {
      setIsAuthenticated(true);
      const payload = decodeToken(token);
      setUsername(payload?.sub || null);
    } else {
      setIsAuthenticated(false);
      setUsername(null);
    }
  }, [pathname]);

  const handleLogout = () => {
    removeToken();
    setIsAuthenticated(false);
    setUsername(null);
    setIsOpen(false);
    router.push("/login");
  };

  const links = isAuthenticated ? protectedLinks : publicLinks;

  // Don't show navbar on login/register if not authenticated
  const isAuthPage = pathname === "/login" || pathname === "/register";

  return (
    <nav
      className={`sticky top-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-gray-950/80 backdrop-blur-xl border-b border-white/5 shadow-lg shadow-black/20"
          : "bg-transparent border-b border-transparent"
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2.5 group">
            <div className="relative">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20 group-hover:shadow-sky-500/40 transition-shadow">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-teal-400 rounded-full pulse-dot" />
            </div>
            <span className="font-bold text-lg text-white tracking-tight">
              Retina<span className="gradient-text">AI</span>
            </span>
          </Link>

          {/* Desktop Nav */}
          <div className="hidden lg:flex items-center space-x-1">
            {links.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`relative px-3.5 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? "text-sky-400"
                      : "text-gray-400 hover:text-gray-200 hover:bg-white/5"
                  }`}
                >
                  {link.label}
                  {isActive && (
                    <motion.div
                      layoutId="navbar-indicator"
                      className="absolute inset-0 bg-sky-500/10 border border-sky-500/20 rounded-lg"
                      style={{ zIndex: -1 }}
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    />
                  )}
                </Link>
              );
            })}

            {/* Auth actions — right side */}
            {isAuthenticated && (
              <div className="flex items-center ml-4 pl-4 border-l border-gray-800 space-x-3">
                {/* User badge */}
                <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-white/5 border border-gray-800">
                  <UserCircle className="w-4 h-4 text-sky-400" />
                  <span className="text-sm font-medium text-gray-300">{username}</span>
                </div>
                {/* Logout button */}
                <button
                  onClick={handleLogout}
                  className="flex items-center space-x-1.5 px-3 py-2 rounded-lg text-sm font-medium text-gray-400 hover:text-red-400 hover:bg-red-500/10 transition-all"
                  title="Sign out"
                >
                  <LogOut className="w-4 h-4" />
                  <span>Logout</span>
                </button>
              </div>
            )}

            {!isAuthenticated && !isAuthPage && (
              <div className="flex items-center ml-4 pl-4 border-l border-gray-800 space-x-2">
                <Link
                  href="/login"
                  className="flex items-center space-x-1.5 px-3.5 py-2 rounded-lg text-sm font-medium text-gray-300 hover:text-white hover:bg-white/5 transition-all"
                >
                  <LogIn className="w-4 h-4" />
                  <span>Sign In</span>
                </Link>
              </div>
            )}
          </div>

          {/* Mobile Toggle */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="lg:hidden p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
            aria-label="Toggle navigation"
          >
            {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="lg:hidden bg-gray-950/95 backdrop-blur-xl border-b border-white/5 overflow-hidden"
          >
            <div className="px-4 py-3 space-y-1">
              {links.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setIsOpen(false)}
                  className={`block px-4 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    pathname === link.href
                      ? "bg-sky-500/10 text-sky-400 border border-sky-500/20"
                      : "text-gray-400 hover:bg-white/5 hover:text-white"
                  }`}
                >
                  {link.label}
                </Link>
              ))}

              {/* Mobile auth actions */}
              {isAuthenticated && (
                <>
                  <div className="border-t border-gray-800 my-2" />
                  <div className="px-4 py-2 flex items-center space-x-2 text-sm text-gray-400">
                    <UserCircle className="w-4 h-4 text-sky-400" />
                    <span>{username}</span>
                  </div>
                  <button
                    onClick={handleLogout}
                    className="w-full text-left px-4 py-2.5 rounded-lg text-sm font-medium text-red-400 hover:bg-red-500/10 transition-colors flex items-center space-x-2"
                  >
                    <LogOut className="w-4 h-4" />
                    <span>Sign out</span>
                  </button>
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}

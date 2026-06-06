"use client";

import { useAuth } from "@/hooks/useAuth";
import { Activity } from "lucide-react";

interface AuthGuardProps {
  children: React.ReactNode;
}

/**
 * Wrapper component that protects routes requiring authentication.
 * Shows a loading spinner while checking auth state, then either
 * renders children (if authenticated) or redirects to /login (via useAuth).
 */
export default function AuthGuard({ children }: AuthGuardProps) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center">
        <div className="relative w-16 h-16 mb-6">
          <div className="absolute inset-0 border-4 border-sky-500/20 rounded-full" />
          <div className="absolute inset-0 border-4 border-sky-500 border-t-transparent rounded-full animate-spin" />
          <div className="absolute inset-0 flex items-center justify-center">
            <Activity className="w-6 h-6 text-sky-400 animate-pulse" />
          </div>
        </div>
        <p className="text-gray-400 text-sm">Verifying authentication...</p>
      </div>
    );
  }

  if (!isAuthenticated) {
    // useAuth will handle the redirect; render nothing while it processes
    return null;
  }

  return <>{children}</>;
}

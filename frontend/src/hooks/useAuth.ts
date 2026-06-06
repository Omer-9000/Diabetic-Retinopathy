"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";
import { getToken, removeToken, isTokenExpired, decodeToken } from "@/lib/auth";

interface AuthState {
  token: string | null;
  user: { username: string; exp: number } | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  logout: () => void;
}

/**
 * Custom hook for authentication state management.
 * Checks localStorage for a valid JWT token. If missing or expired,
 * redirects to /login.
 */
export function useAuth(): AuthState {
  const router = useRouter();
  const pathname = usePathname();
  const [token, setTokenState] = useState<string | null>(null);
  const [user, setUser] = useState<{ username: string; exp: number } | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const logout = useCallback(() => {
    removeToken();
    setTokenState(null);
    setUser(null);
    router.push("/login");
  }, [router]);

  useEffect(() => {
    const storedToken = getToken();

    if (!storedToken || isTokenExpired(storedToken)) {
      // Don't redirect if already on login or register
      if (pathname !== "/login" && pathname !== "/register") {
        removeToken();
        window.location.href = "/login";
      } else {
        setIsLoading(false);
      }
      return;
    }

    const payload = decodeToken(storedToken);
    setTokenState(storedToken);
    setUser(
      payload
        ? { username: payload.sub, exp: payload.exp }
        : null
    );
    setIsLoading(false);
  }, [pathname]);

  return {
    token,
    user,
    isAuthenticated: !!token && !isLoading,
    isLoading,
    logout,
  };
}

import { createContext, useContext, useState, useEffect, type ReactNode } from 'react';
import { authApi, isAuthenticated, clearTokens } from '@/lib/api';
import type { User, SetupStatus } from '@/types';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isSetupRequired: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => void;
  checkAuth: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSetupRequired, setIsSetupRequired] = useState(false);

  const checkAuth = async () => {
    setIsLoading(true);
    try {
      // First check if setup is required
      const setupStatus: SetupStatus = await authApi.checkSetup();
      setIsSetupRequired(setupStatus.setup_required);

      // If authenticated, try to get user info
      if (isAuthenticated()) {
        try {
          const currentUser = await authApi.getCurrentUser();
          setUser(currentUser);
        } catch {
          // Token might be invalid, clear it
          clearTokens();
          setUser(null);
        }
      } else {
        setUser(null);
      }
    } catch (error) {
      console.error('Auth check failed:', error);
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (username: string, password: string) => {
    await authApi.login({ username, password });
    await checkAuth();
  };

  const register = async (username: string, password: string) => {
    await authApi.register({ username, password });
    await checkAuth();
  };

  const logout = () => {
    clearTokens();
    setUser(null);
    window.location.href = '/login';
  };

  useEffect(() => {
    checkAuth();
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isSetupRequired,
        login,
        register,
        logout,
        checkAuth,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

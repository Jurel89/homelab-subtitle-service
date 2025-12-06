import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  PlusCircle,
  Settings,
  FileText,
  BarChart3,
  BookOpen,
  LogOut,
  Menu,
  X,
} from 'lucide-react';
import { useState } from 'react';
import { cn } from '@/lib/utils';
import { useAuth } from '@/context/AuthContext';
import { Button } from '@/components/ui/Button';

interface NavItem {
  name: string;
  href: string;
  icon: React.ReactNode;
}

const navItems: NavItem[] = [
  { name: 'Dashboard', href: '/', icon: <LayoutDashboard className="h-5 w-5" /> },
  { name: 'New Job', href: '/new', icon: <PlusCircle className="h-5 w-5" /> },
  { name: 'Logs', href: '/logs', icon: <FileText className="h-5 w-5" /> },
  { name: 'KPIs', href: '/kpis', icon: <BarChart3 className="h-5 w-5" /> },
  { name: 'Settings', href: '/settings', icon: <Settings className="h-5 w-5" /> },
  { name: 'Wiki', href: '/wiki', icon: <BookOpen className="h-5 w-5" /> },
];

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();
  const { user, logout } = useAuth();

  return (
    <>
      {/* Mobile menu button */}
      <button
        className="fixed left-4 top-4 z-50 rounded-md bg-card p-2 lg:hidden"
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
      </button>

      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          'fixed left-0 top-0 z-40 h-screen w-64 transform border-r bg-card transition-transform duration-200 ease-in-out lg:translate-x-0',
          isOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <div className="flex h-full flex-col">
          {/* Logo */}
          <div className="flex h-16 items-center border-b px-6">
            <Link to="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold text-primary">SUBSVC</span>
            </Link>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-1 px-3 py-4">
            {navItems.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setIsOpen(false)}
                  className={cn(
                    'flex items-center space-x-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-primary text-primary-foreground'
                      : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                  )}
                >
                  {item.icon}
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>

          {/* User section */}
          <div className="border-t p-4">
            <div className="mb-3 flex items-center space-x-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground">
                {user?.username?.[0]?.toUpperCase() || 'U'}
              </div>
              <div className="flex-1 truncate">
                <p className="truncate text-sm font-medium">{user?.username || 'User'}</p>
                <p className="text-xs text-muted-foreground">
                  {user?.is_admin ? 'Admin' : 'User'}
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-start"
              onClick={logout}
            >
              <LogOut className="mr-2 h-4 w-4" />
              Logout
            </Button>
          </div>
        </div>
      </aside>
    </>
  );
}

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './presentation/hooks/useAuth';
import { UserUseCase } from './domain/usecases/user.usecase';
import { UserRepositoryImpl } from './data/models/user.repository.impl';
import MainLayout from './presentation/layout/MainLayout';
import Dashboard from './presentation/pages/Dashboard';
import UsersManagement from './presentation/pages/UsersManagement';
import ExerciseManagement from './presentation/pages/ExerciseManagement';

// Create a simple login page for development
const LoginPage: React.FC = () => {
  const [email, setEmail] = React.useState('');
  const [password, setPassword] = React.useState('');
  const [error, setError] = React.useState('');
  const [loading, setLoading] = React.useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      // For development, you can use test credentials or bypass login
      if (email === 'admin@admin.com' && password === 'admin') {
        // Store a mock token for development
        localStorage.setItem('accessToken', 'mock-admin-token');
        localStorage.setItem('refreshToken', 'mock-refresh-token');
        window.location.reload();
      } else {
        setError('Invalid credentials. Use admin@admin.com / admin for development');
      }
    } catch (error: any) {
      setError(error.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <div className="flex justify-center">
          <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-lg">OM</span>
          </div>
        </div>
        <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
          OmniMer Admin
        </h2>
        <p className="mt-2 text-center text-sm text-gray-600">
          Sign in to your admin account
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10">
          <form className="space-y-6" onSubmit={handleLogin}>
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                Email address
              </label>
              <div className="mt-1">
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  placeholder="admin@admin.com"
                />
              </div>
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                Password
              </label>
              <div className="mt-1">
                <input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="current-password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  placeholder="admin"
                />
              </div>
            </div>

            {error && (
              <div className="text-sm text-red-600">{error}</div>
            )}

            <div>
              <button
                type="submit"
                disabled={loading}
                className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Signing in...' : 'Sign in'}
              </button>
            </div>

            <div className="mt-6 text-sm text-gray-600">
              <p className="font-medium">Development Credentials:</p>
              <p>Email: admin@admin.com</p>
              <p>Password: admin</p>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const token = localStorage.getItem('accessToken');

  // Simple check - in production you'd want more robust validation
  if (!token) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};

const App: React.FC = () => {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          {/* Login route */}
          <Route path="/login" element={<LoginPage />} />

          {/* Protected routes */}
          <Route path="/" element={
            <ProtectedRoute>
              <MainLayout />
            </ProtectedRoute>
          }>
            {/* Default redirect */}
            <Route index element={<Navigate to="/dashboard" replace />} />

            {/* Dashboard */}
            <Route path="dashboard" element={<Dashboard />} />

            {/* User Management */}
            <Route path="users" element={<UsersManagement />} />
            <Route path="users/roles" element={<div className="p-6"><h1 className="text-2xl font-bold">Role Management</h1><p className="text-gray-600">Manage user roles and permissions</p></div>} />
            <Route path="users/permissions" element={<div className="p-6"><h1 className="text-2xl font-bold">Permission Management</h1><p className="text-gray-600">Manage system permissions</p></div>} />

            {/* Exercise Management */}
            <Route path="exercises" element={<ExerciseManagement />} />
            <Route path="exercises/equipment" element={<div className="p-6"><h1 className="text-2xl font-bold">Equipment Management</h1><p className="text-gray-600">Manage exercise equipment</p></div>} />
            <Route path="exercises/body-parts" element={<div className="p-6"><h1 className="text-2xl font-bold">Body Parts Management</h1><p className="text-gray-600">Manage exercise body parts</p></div>} />
            <Route path="exercises/muscles" element={<div className="p-6"><h1 className="text-2xl font-bold">Muscles Management</h1><p className="text-gray-600">Manage muscle groups</p></div>} />
            <Route path="exercises/types" element={<div className="p-6"><h1 className="text-2xl font-bold">Exercise Types Management</h1><p className="text-gray-600">Manage exercise types</p></div>} />
            <Route path="exercises/categories" element={<div className="p-6"><h1 className="text-2xl font-bold">Exercise Categories Management</h1><p className="text-gray-600">Manage exercise categories</p></div>} />
            <Route path="exercises/list" element={<div className="p-6"><h1 className="text-2xl font-bold">Exercises Management</h1><p className="text-gray-600">Manage exercises</p></div>} />

            {/* Catch all */}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Route>

          {/* Root redirect */}
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
};

export default App;

import { darkTheme } from '@/presentation/theme/dark';
import { lightTheme } from '@/presentation/theme/light';
import React, { createContext, useContext, useState, ReactNode } from 'react';

export type ThemeMode = 'light' | 'dark';

interface ThemeContextType {
  theme: typeof lightTheme;
  mode: ThemeMode;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType>({
  theme: lightTheme,
  mode: 'light',
  toggleTheme: () => {},
});

export const ThemeProvider = ({ children }: { children: ReactNode }) => {
  const [mode, setMode] = useState<ThemeMode>('light');

  const toggleTheme = () => {
    setMode(prev => (prev === 'light' ? 'dark' : 'light'));
  };

  const theme = mode === 'light' ? lightTheme : darkTheme;

  return (
    <ThemeContext.Provider value={{ theme, mode, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => useContext(ThemeContext);

import api from './api';
import { User, AuthTokens } from '../types';

export const authService = {
  async login(email: string, password: string): Promise<AuthTokens> {
    const { data } = await api.post<AuthTokens>('/auth/login', { email, password });
    localStorage.setItem('access_token', data.access_token);
    return data;
  },

  async register(email: string, password: string, fullName: string): Promise<AuthTokens> {
    const { data } = await api.post<AuthTokens>('/auth/register', {
      email, password, full_name: fullName,
    });
    localStorage.setItem('access_token', data.access_token);
    return data;
  },

  async getMe(): Promise<User> {
    const { data } = await api.get<User>('/auth/me');
    return data;
  },

  logout(): void {
    localStorage.removeItem('access_token');
    window.location.href = '/login';
  },

  isAuthenticated(): boolean {
    return !!localStorage.getItem('access_token');
  },
};

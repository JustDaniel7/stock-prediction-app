import React from 'react';
import { Inter } from 'next/font/google';
import './globals.css';
import Header from './components/Header';
import Footer from './components/Footer';

const inter = Inter({ subsets: ['latin'] });

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="en" className={inter.className}>
        <body className="flex flex-col min-h-screen bg-gradient-to-b from-gray-50 to-gray-100">
        <Header />
        <main className="flex-grow container mx-auto px-4 py-8 sm:px-6 lg:px-8">
            <div className="bg-white shadow-xl rounded-lg p-6">
                {children}
            </div>
        </main>
        <Footer />
        </body>
        </html>
    );
}
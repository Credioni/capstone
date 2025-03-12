// src/App.tsx
import './App.css'
import React, { useEffect } from 'react';
import { useLynxGlobalEventListener } from '@lynx-js/react'
import { FrontPage } from "./pages/FrontPage.tsx";


export function App() {
    useEffect(() => {
        // Check if we're in the browser environment
        if (typeof window !== "undefined") {
          const handleKeyPress = (event) => {
            console.log('Key pressed:', event.key);
            // Additional logic to handle the key press
          };

          // Add event listener for keydown events
          window.addEventListener('keydown', handleKeyPress);

          // Clean up the event listener on component unmount
          return () => {
            window.removeEventListener('keydown', handleKeyPress);
          };
        }
      }, []);

    return (
        <view>
            <view className='App'>
                <view className='Background' />
                <view className="App">
                    <FrontPage />
                </view>
            </view>
        </view>
    );
}


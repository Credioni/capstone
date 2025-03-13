// src/App.tsx
import './App.css'
import { useLynxGlobalEventListener, useState, useMainThreadRef } from '@lynx-js/react'
import { MainThread } from "@lynx-js/types"

import { FrontPage } from "./pages/FrontPage.tsx";
import { MainThreadDraggable } from "./MainThreadTriggers.tsx";

// { /* WEB ELEMENTS */}
// import '@lynx-js/web-core';
// import '@lynx-js/web-core/index.css';
// const handleKeyDown = (event:any) => {
//     console.log(`Key pressed: ${event.key}`);
//   };
// document.addEventListener("keydown", handleKeyDown);


export function App() {
    let eleRef = useMainThreadRef<MainThread.Element>();

    function handleTapMainThread() {
      'main thread';
      eleRef.current?.setStyleProperty('height', '30px');
      console.log("MainThread Tap");
    }

    return (
        <view className='App' main-thread:bindTap={handleTapMainThread}>
            <view />
            { /*
            <MainThreadDraggable size={100} />
            */ }
            <view className='Background' />
            <view className="App">
                <FrontPage />
            </view>
            <view className='footer'>
                <text> Footer </text>
            </view>
        </view>
    );
}


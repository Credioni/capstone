import React, { useState } from 'react';
import "./HomePage.css"
import SearchBar from './homepage/SearchBar';
import QuicLinks from './homepage/QuickLinks';


function HomePage() {
    return (
        <main className="homepage-content">
            <div className="logo-area">
                <h1 className="logo">Welcome to ArXiv RAG Search</h1>
            </div>
            <SearchBar />
            <QuicLinks />
        </main>
    );
}

export default HomePage;
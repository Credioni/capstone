import React, { useState } from 'react';
import "./HomePage.css"
import SearchBar from '../components/SearchBar';
import QuicLinks from '../components/QuickLinks';
//import SearchIcon from "./imgs/search_icon.svg?react'";


function HomePage() {
    return (
        <main className="homepage-content">
            <div className="logo-area">
                <h1 className="logo">Welcome to ArXiv API</h1>
            </div>
            <SearchBar />
            <QuicLinks />
        </main>
    );
}

export default HomePage;
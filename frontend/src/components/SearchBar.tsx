import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import "./SearchBar.css";


export default function SearchBar() {
    const navigate = useNavigate();
    const [searchQuery, setSearchQuery] = useState('');

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault();
        navigate(`/search?q=${encodeURIComponent(searchQuery)}`);
    };

    return (
        <div className="search-container">
            <form className="search-form" onSubmit={handleSearch}>
                <input
                    className="search-input"
                    type="text"
                    placeholder="Search..."
                    value={searchQuery}
                    onChange={(e:any) => setSearchQuery(e.target.value)}
                />

                {/* <Button variant="contained" onClick={handleSearch}>Search</Button> */}
            </form>
        </div>
    )
}
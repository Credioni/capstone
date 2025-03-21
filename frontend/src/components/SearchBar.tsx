import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import "./SearchBar.css";

type MediaFile = {
    file: File;
    preview: string;
    type: string;
};

export default function SearchBar() {
    const navigate = useNavigate();
    const [searchQuery, setSearchQuery] = useState('');
    const [mediaFiles, setMediaFiles] = useState<MediaFile[]>([]);
    const [isDragging, setIsDragging] = useState(false);
    const [showMediaPreview, setShowMediaPreview] = useState(false);
    const [isSearching, setIsSearching] = useState(false);

    const fileInputRef = useRef<HTMLInputElement>(null);

    // Handle search submission
    const handleSearch = async (e: React.FormEvent) => {
        e.preventDefault();

        // Set loading state
        setIsSearching(true);

        try {
            // Create a FormData object to send both query and files
            const formData = new FormData();

            // Always add the query to FormData, regardless of whether there are files
            formData.append('query', searchQuery);

            // Add all media files to the form data if they exist
            mediaFiles.forEach((mediaFile, index) => {
                formData.append('media', mediaFile.file);
            });

            console.log("formData \n", formData)
            // Send the request to the backend using one endpoint for everything
            const response = await fetch('http://localhost:8080/query', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Search request failed');
            }

            // Parse the response
            const data = await response.json();

            // Navigate to results page with the search results or ID
            if (data.query_id) {
                navigate(`/search?id=${data.query_id}`);
            } else {
                // Alternative: you might get direct results or handle differently
                // For now, just show the results page with the query as a fallback
                navigate(`/search?q=${encodeURIComponent(searchQuery)}`);
            }

        } catch (error) {
            console.error('Error performing search:', error);
            // Handle error - you could show an error message to the user
            alert('There was an error performing your search. Please try again.');
        } finally {
            setIsSearching(false);
        }
    };

    // Handle file selection via input button
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            addFiles(Array.from(e.target.files));
            setShowMediaPreview(true);
        }
    };

    // Process and add files to state
    const addFiles = (files: File[]) => {
        const newFiles: MediaFile[] = [];

        files.forEach(file => {
            // Determine file type
            let type = 'other';
            if (file.type.startsWith('image/')) {
                type = 'image';
            } else if (file.type.startsWith('audio/')) {
                type = 'audio';
            } else if (file.type.startsWith('video/')) {
                type = 'video';
            }

            // Create preview URL
            const preview = URL.createObjectURL(file);
            newFiles.push({ file, preview, type });
        });

        setMediaFiles(prev => [...prev, ...newFiles]);
    };

    // Handle drag events
    const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    };

    const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            addFiles(Array.from(e.dataTransfer.files));
            setShowMediaPreview(true);
        }
    };

    // Remove a file from the list
    const removeFile = (index: number) => {
        setMediaFiles(prev => {
            const newFiles = [...prev];
            URL.revokeObjectURL(newFiles[index].preview); // Clean up the URL object
            newFiles.splice(index, 1);

            // Hide preview if no files left
            if (newFiles.length === 0) {
                setShowMediaPreview(false);
            }

            return newFiles;
        });
    };

    // Clean up object URLs when component unmounts
    React.useEffect(() => {
        return () => {
            mediaFiles.forEach(mediaFile => {
                URL.revokeObjectURL(mediaFile.preview);
            });
        };
    }, []);

    return (
        <div
            className={`search-container ${isDragging ? 'dragging' : ''}`}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
        >
            <form className="search-form" onSubmit={handleSearch}>
                <div className="search-input-container">
                    <input
                        className={`search-input ${showMediaPreview ? 'with-media' : ''}`}
                        type="text"
                        placeholder="Search..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                    />

                    {/* Preview of uploaded media inside search bar */}
                    {showMediaPreview && (
                        <div className="media-preview-inline">
                            {mediaFiles.map((media, index) => (
                                <div key={index} className="media-item-inline">
                                    {media.type === 'image' && (
                                        <div className="thumbnail">
                                            <img src={media.preview} alt={`Uploaded ${index}`} />
                                        </div>
                                    )}
                                    {media.type === 'audio' && (
                                        <div className="thumbnail audio-thumb">
                                            <span>🎵</span>
                                        </div>
                                    )}
                                    {media.type === 'video' && (
                                        <div className="thumbnail video-thumb">
                                            <span>🎬</span>
                                        </div>
                                    )}
                                    <button
                                        type="button"
                                        className="remove-button"
                                        onClick={() => removeFile(index)}
                                    >
                                        ×
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                <div className="search-actions">
                    <button
                        type="button"
                        className="multimedia-button"
                        onClick={() => fileInputRef.current?.click()}
                    >
                        Add Multimedia
                    </button>
                    <button
                        type="submit"
                        className="search-button"
                        disabled={isSearching}
                    >
                        {isSearching ? 'Searching...' : 'Search'}
                    </button>
                </div>

                {/* Hidden file input */}
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*,audio/*,video/*"
                    style={{ display: 'none' }}
                    onChange={handleFileChange}
                    multiple
                />

                {/* Drag and drop instruction */}
                {isDragging && (
                    <div className="drop-zone-overlay">
                        <p>Drop your files here</p>
                    </div>
                )}
            </form>
        </div>
    );
}

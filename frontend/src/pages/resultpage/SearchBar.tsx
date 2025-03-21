import React, { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
    Container,
    Typography,
    Box,
    TextField,
    InputAdornment,
    IconButton,
    Divider,
    Paper,
} from '@mui/material';

export function SearchBar({searchQuery, setSearchQuery, handleSearch, ...props}: any){
    return (
        <Paper
            {...props}
            component="form"
            onSubmit={handleSearch}
            elevation={0}
        >
            <TextField
                fullWidth
                placeholder="Search articles..."
                value={searchQuery}
                variant="outlined"
                size="small"
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        console.log("Hello");
                        handleSearch(e);
                    }
                }}

            />
        </Paper>
    );
}
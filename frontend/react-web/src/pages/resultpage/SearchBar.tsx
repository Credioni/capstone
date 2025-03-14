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
import SearchIcon from '@mui/icons-material/Search';


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
                onChange={(e) => setSearchQuery(e.target.value)}
                variant="outlined"
                size="small"
                // InputProps={{
                //     endAdornment: (
                //     <InputAdornment position="end">
                //         <IconButton type="submit" edge="end">
                //         <SearchIcon />
                //         </IconButton>
                //     </InputAdornment>
                //     ),
                // }}
            />
        </Paper>
    );
}
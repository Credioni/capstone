import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
    Container,
    Typography,
    Box,
    TextField,
    InputAdornment,
    IconButton,
    Divider,
    Paper,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    List,
    ListItem
  } from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import SearchIcon from '@mui/icons-material/Search';
import { SampleResults } from '../assets/SampleResults';
import ResultItem from './resultpage/ResultItem';
import {SearchBar} from "./resultpage/SearchBar"



function ResultPage() {
    const [searchParams] = useSearchParams();
    const query = searchParams.get('q');
    const navigate = useNavigate();
    const [results, setResults] = useState(query?.includes("test") ? SampleResults:[]);
    const [category, setCategory] = useState('all');
    const [searchQuery, setSearchQuery] = useState(query);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        setSearchQuery(query);

        fetchResults(query);
    }, [query]);

    // Function to fetch results based on query
    const fetchResults = async (searchTerm) => {
        setLoading(true);
        try {
            // For demo purposes, check if contains "test"
            if (searchTerm?.includes("test")) {
                setResults(SampleResults);
            } else if (searchTerm) {
                // In a real app, you would fetch data from an API
                // const response = await api.search(searchTerm, category);
                // setResults(response.data);

                // Simulating API call with timeout
                setTimeout(() => {
                    setResults([]);
                    setLoading(false);
                }, 500);
            } else {
                setResults([]);
            }
        } catch (error) {
            console.error('Error fetching results:', error);
            setResults([]);
        } finally {
            setLoading(false);
        }
    };

    const navHome = (event) => {
        event.preventDefault();
        navigate(`/`);
    }

    const handleSearch = (event) => {
        event.preventDefault();
        // In a real app, you would fetch results based on searchQuery
        console.log('Searching for:', searchQuery);
        navigate(`/search?q=${encodeURIComponent(searchQuery || "")}`);
    };

      const handleResultClick = (id) => {
        console.log('Clicked on result:', id);
        // Navigate to detail page or perform other actions
    };

    return (
        <div className='w-dvh min-h-dvh justify-items-center'>
            <header className='w-full h-20'>
                {/* Header control row */}
                <div className='h-full grid grid-cols-3 place-items-center  bg-[#2c243c]'>
                    <Typography
                        component="button"
                        sx={{color: "white"}}
                        className="justify-self-center"
                        variant="h5"
                        gutterBottom
                        onClick={navHome}
                    >
                        ArXiv RAG Search
                    </Typography>

                    <SearchBar
                        className="w-full"
                        searchQuery={searchQuery}
                        setSearchQuery={setSearchQuery}
                        handleSearch={handleSearch}
                    />
                    {/* Right header - Info*/}
                    <div className='flex justify-self-start m-5'>
                        <Divider className='bg-gray-900' orientation='vertical'/>
                        <HelpOutlineIcon/>
                        <Typography> Syntax </Typography>
                    </div>
                </div>
            </header>

            {/* Result Bar Info */}
            <div className='h-10 w-full grid grid-cols-3 place-items-cente'>
                <div/>
                {/* <FormControl sx={{ minWidth: 150 }} size="small">
                    <InputLabel id="category-select-label">Category</InputLabel>
                    <Select
                        labelId="category-select-label"
                        value={category}
                        label="Category"
                        onChange={(e) => setCategory(e.target.value)}
                    >
                        <MenuItem value="all">All categories</MenuItem>
                        <MenuItem value="quant-ph">Quantum Physics</MenuItem>
                        <MenuItem value="physics.app-ph">Applied Physics</MenuItem>
                        <MenuItem value="cs.LG">Machine Learning</MenuItem>
                        <MenuItem value="physics.optics">Optics</MenuItem>
                    </Select>
                </FormControl> */}
                {/* Results count */}
                <Box className="justify-self-left ml-(50%)" sx={{ mb: 2 }}>
                    <Typography variant="subtitle1" color="text.secondary">
                        Found {results.length} results
                    </Typography>
                </Box>
            </div>

            {/* Results */}
            <Box className="grid-flow-col grid-cols-${results.length} justify-items-center pt-5 max-w-5xl">
                <List sx={{ listStyle: "decimal", pl: 4 }}>
                    {results.length > 0 ? (
                        results.map((result, index) => (
                            <ResultItem
                                // className="max-w-25 pb-2"
                                key={index}
                                index={index}
                                result={result}
                            />
                    ))
                    ) : (
                        <Typography align="center" sx={{ mt: 4 }}>
                            No results found. Try changing your search terms.
                        </Typography>
                    )}
                </List>
            </Box>

            {/* Page Selector */}
        </div>
    );
}

export default ResultPage;

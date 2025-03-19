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
    CardContent,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    List,
    ListItem,
    Button,
    Card
  } from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import SearchIcon from '@mui/icons-material/Search';
import { SampleResults } from '../assets/SampleResults';
import ResultItem from './resultpage/ResultItem';
import TypeWriter from './resultpage/ResultItem';
import {SearchBar} from "./resultpage/SearchBar"
import { FetchData } from '../services/RagApi';
import TypewriterText from './resultpage/TypeWriter';
/* EXAMPLE response
{
  "status": "success",
  "answer": "Quantum computing is...",
  "sources": [
    {
      "id": "1902.12345",
      "title": "Fast-response low power atomic oven for integration into an ion microchip",
      "authors": ["Vijay Kumar", "Martin Siegele-Brown", "Parsa Rahimi", "Matthew Aylett", "Sebastian Weidt", "Winfried Karl Hensinger"],
      "abstract": "We present a novel microfabricated neutral atom source...",
      "categories": ["physics.app-ph", "quant-ph"],
      "pages": 5,
      "figures": 5,
      "score": 0.85,
      "type": "text",
      "preview": "The microfabricated neutral atom source is..."
    }
  ],
  "query": "atomic oven microchip",
  "images": [...]
}
*/


function ResultPage() {
    const [searchParams] = useSearchParams();
    const query = searchParams.get('q');
    const navigate = useNavigate();


    const [category, setCategory] = useState('all');
    const [searchQuery, setSearchQuery] = useState(query);
    const [loading, setLoading] = useState(false);

    // RESPONSES
    // raw
    const [rawresponse, setRawResponse] = useState(query?.includes("test") ? SampleResults:null);
    // generated answer
    const [answer, setAnswer]   = useState(query?.includes("test") ? "Hello world!":null);
    // rag sources
    const [sources, setSources] = useState(query?.includes("test") ? SampleResults:[]);
    // results
    const [results, setResults] = useState(query?.includes("test") ? SampleResults:[]);



    useEffect(() => {
        setSearchQuery(query);
        fetchResults(searchQuery);
    }, [query]);

    function GenerateResultsAndAnswer(raw_response) {
        const sources: [] = raw_response.sources;

        let parsed_results = [];
        for (let i = 0; i < sources.length; i++) {
            const tmp: any = sources[i];
            const asd = {
                id: tmp.paper_id,
                title: "Fast-response low power atomic oven for integration into an ion microchip",
                authors: ["Vijay Kumar", "Martin Siegele-Brown", "Parsa Rahimi", "Matthew Aylett", "Sebastian Weidt", "Winfried Karl Hensinger"],
                abstract: "We present a novel microfabricated neutral atom source for quantum technologies that can be easily integrated onto microchip devices using well-established MEMS fabrication techniques, and contrast this to conventional off-chip ion loading mechanisms. The heating filament of the device is shown to be as small as 90×90 μm².",
                categories: ["physics.app-ph", "quant-ph"],
                // contentTypes: ["pdf", "images", "equations"],
                // submissionDate: "13 March, 2025",
                figures: 5,
                pages: 5
            };
            parsed_results.push(asd);
        }
        setAnswer(raw_response.answer);
        setResults(parsed_results)
    }

    // Function to fetch results based on query
    async function fetchResults(searchTerm) {
        console.log("Fetching result");

        setLoading(true);
        try {
            if (searchTerm?.includes("test")) {
                // For demo purposes, check if contains "test"
                setResults(SampleResults);
                setRawResponse(null);
            } else if (searchTerm) {
                let path = `http://localhost:8080/query?q=${encodeURIComponent(query || "")}`;
                const response = await FetchData(path);
                console.log("response", response)
                setRawResponse(response);

                setSources(response.sources);
                GenerateResultsAndAnswer(response);

                // Simulating API call with timeout
                setTimeout(() => {
                    setRawResponse(null);
                    setLoading(false);
                }, 500);
            } else {
                setRawResponse(null);
            }
        } catch (error) {
            console.error('Error fetching results:', error);
            setRawResponse(null);
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
        console.log('Searching for:', searchQuery);
        // Zeroing previous response
        setRawResponse(null);
        setAnswer(null);
        setSources([]);
        setResults([]);
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

                        <Button onClick={() => GenerateResultsAndAnswer(rawresponse)}>
                            Query
                        </Button>
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
                {/* <Box className="justify-self-left ml-(50%)" sx={{ mb: 2 }}>
                    <Typography variant="subtitle1" color="text.secondary">
                        Found {results?.length | 0 } results
                    </Typography>
                </Box> */}
            </div>

            {/* Results */}
            <Box className="grid-flow-col grid-cols-${results.length} justify-items-center pt-5 max-w-5xl">
                {/* Generated RAG LLM answer */}
                <TypewriterText text={answer} typingSpeed={5} className="min-w-fit"/>

                <List sx={{ listStyle: "decimal", pl: 4 }}>
                    {results?.length > 0 ? (
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

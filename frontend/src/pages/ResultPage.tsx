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
import ResultArxivItem from './resultpage/ResultArxivItem';
import TypeWriter from './resultpage/ResultArxivItem';
import {SearchBar} from "./resultpage/SearchBar"
import { FetchData } from '../services/RagApi';
import TypewriterText from './resultpage/TypeWriter';
import ResultItemVideo from './resultpage/ResultItemVideo';

interface PaperMetadata {
    title: string;
    abstract: string;
    published: string;
    authors: string[];
    doi: string;
    url: string;
}

interface PaperInformation {
    paper_id: string;
    score: number;
    images: any[];
    metadata: PaperMetadata;
}

interface ResultText {
    id: string;
    score: number;
    text: string;
    title: string
}

interface ResultImage {
    id: string;
    caption: string;
    path: string;
    score: number;
    source: string;
}

interface ResultYoutube {
    author: string;
    score: number;
    title: string;
    // etc
}


function ResultPage() {
    const [searchParams] = useSearchParams();
    const id = searchParams.get('id');
    const navigate = useNavigate();


    const [category, setCategory] = useState('all');
    const [searchQuery, setSearchQuery] = useState(id);
    const [loading, setLoading] = useState(false);

    // RESPONSES
    // raw
    const [rawresponse, setRawResponse] = useState(null);
    // generated answer
    const [answer, setAnswer]   = useState(null);
    // faiss results
    const [results, setResults] = useState([]);



    useEffect(() => {
        setSearchQuery(id);
        fetchResults(searchQuery);
    }, [id]);

    function GenerateResultsAndAnswer(raw_response) {
        // const sources: [] = raw_response.sources;

        // let parsed_sources: any[] = [];
        // for (let i = 0; i < sources.length; i++) {
        //     const source: any = sources[i];
        //     const metadata: any = source.metadata;

        //     const asd: any = {
        //         score: source.score,
        //         id: source.paper_id,
        //         title: metadata.title,
        //         authors: metadata.authors,
        //         abstract: metadata.abstract,
        //         url: metadata.url,
        //         // categories: metadata,
        //         // contentTypes: ["pdf", "images", "equations"],
        //         // submissionDate: "13 March, 2025",
        //         figures: 5,
        //         pages: 5
        //     };
        //     parsed_sources.push(asd);
        // }
        // setAnswer(raw_response.answer);
        // setResults(parsed_sources)
    }

    // Function to fetch results based on query
    async function fetchResults(searchTerm) {
        console.log("id", id);
        setLoading(true);
        try {
            if (searchTerm) {
                let path = `http://localhost:8080/result?q=${encodeURIComponent(id || "")}`;
                const response = await FetchData(path);
                console.log("response", response)
                setRawResponse(response);
                setResults(response.results)

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
                <div className='h-full grid grid-cols-2 place-items-center  bg-[#2c243c]'>
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

                    {/* <SearchBar
                        className="w-full"
                        searchQuery={searchQuery}
                        setSearchQuery={setSearchQuery}
                        handleSearch={handleSearch}
                    /> */}
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

            {/* Page Content */}
            <Box>
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

                {/* ArXiv Results */}
                <Box className="grid-flow-col grid-cols-${results.length} justify-items-center pt-5 max-w-5xl">
                    {/* Generated RAG LLM answer */}
                    <TypewriterText text={answer} typingSpeed={5} className="min-w-fit"/>

                    <List sx={{ listStyle: "decimal", pl: 4 }}>
                        { results?.text?.length > 0 ? (
                            results.text.map((paper, index) => (
                                <ResultArxivItem
                                    // className="max-w-25 pb-2"
                                    key={index}
                                    index={index}
                                    result={paper}
                                />
                            ))
                            ) : (
                                <Typography align="center" sx={{ mt: 4 }}>
                                    No results found. Try changing your search terms.
                                </Typography>
                        )}
                    </List>
                </Box>

                {/* Images and Audio */}
                <Box className="max-w-5xl w-full min-w-full justify-items-center">
                    <Card className="w-full">
                        <Typography>
                            { "Images and audio content" }
                        </Typography>
                    </Card>
                </Box>

                {/* Youtube Videos */}
                <Box className="grid-flow-col max-w-5xl w-full min-w-full justify-items-center pt-5">

                    <Typography className='w-fit justify-self-start' variant="h6">
                        {"You may find intrest in these Scientific Content Creators..."}
                    </Typography>


                    <List sx={{ listStyle: "decimal", pl: 4 }}>
                        { results?.video?.length > 0 ? (
                            results.video.map((video, index) => (
                                <ResultItemVideo video={video} className="min-w-fit"/>
                            ))
                        ): ""}
                    </List>
                </Box>
            </Box>
        </div>
    );
}

export default ResultPage;

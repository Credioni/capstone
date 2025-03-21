import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
    Typography,
    Box,
    Divider,
    List,
    Card,
    CircularProgress
  } from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import ShuffleIcon from '@mui/icons-material/Shuffle';
import FormatAlignJustifyIcon from '@mui/icons-material/FormatAlignJustify';
import { SampleResults } from '../assets/SampleResults';
import ResultArxivItem from './resultpage/ResultArxivItem';
import { FetchData } from '../services/RagApi';
import TypewriterText from './resultpage/TypeWriter';
import ResultItemVideo from './resultpage/ResultItemVideo';
import ImageDisplay from './resultpage/ImageDisplay';
import AudioDisplay from './resultpage/AudioDisplay';
import LoadingText from './resultpage/LoadingText';

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
const LOADING_MSG = [
    "Searching response over 2.5 million reasearch papers...",
    "Searching throught various multimedia sources...",
]

const RAG_LOADING_MSG = [
    "Quering over 2.5 million research articles...",
    "Deepseek R1 thinking about your query...",
    "Generating reponses based on content below...",
    "Generating answer based on reasearch articles...",
]


function ResultPage() {
    const [searchParams] = useSearchParams();
    const id = searchParams.get('id');
    const navigate = useNavigate();


    const [category, setCategory] = useState('all');
    const [searchQuery, setSearchQuery] = useState(id);
    const [loading, setLoading] = useState(true);

    // RESPONSES
    // raw
    const [rawresponse, setRawResponse] = useState(null);
    // generated answer
    const [answer, setAnswer]   = useState(null);
    // faiss results
    const [results, setResults] = useState([]);


    useEffect(() => {
        if (rawresponse?.answer != null) {
            setAnswer(rawresponse.answer)
        }
    }, [rawresponse]);


    useEffect(() => {
        setSearchQuery(id);
        fetchResults(searchQuery);
    }, [id]);

    // Function to fetch results based on query
    async function fetchResults(searchTerm) {
        console.log("id", id);

        setLoading(true);
        try {
            if (searchTerm) {
                let path = `http://localhost:8080/result?q=${encodeURIComponent(id || "")}`;
                const response = await FetchData(path);
                console.log("response", response)

                // Set response content
                setRawResponse(response);
                setResults(response.results)

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
                <div className='h-full grid grid-cols-3 place-items-center  bg-[#2c243c]'>
                    <div/>
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

                    {/* Right header - Info*/}
                    <div className='flex justify-self-start m-5'>
                        <FormatAlignJustifyIcon sx={{color: "white"}} />
                        <Typography sx={{color: "white"}}> Advanced </Typography>

                        <ShuffleIcon className="ml-5" sx={{color: "white"}} />
                        <Typography sx={{color: "white"}}> Random </Typography>

                        <HelpOutlineIcon className="ml-5" sx={{color: "white"}} />
                        <Typography sx={{color: "white"}}> Syntax </Typography>
                    </div>
                </div>
            </header>

            {/* Page Content */}

            {/* Generated RAG LLM answer */}
            <Box className="pt-5 w-full h-full justify-items-center ">
                { answer !== null ?
                    <TypewriterText text={answer} typingSpeed={5} className="min-w-fit"/>
                :
                    <Box className="flex flex-col items-center justify-center w-full h-screen pb-56">
                        <CircularProgress color="success" />
                        <LoadingText list={answer === null ? LOADING_MSG : RAG_LOADING_MSG} />
                    </Box>
                }
            </Box>

            <Box sx={{ display: loading ? "none": "block" }} >
                {/* ArXiv Results */}
                <Box
                    className="grid-flow-col grid-cols-${results.length} justify-items-center pt-5 max-w-5xl"
                    sx={{ display: results?.text == null ? "none": "block" }}
                >
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
                <Card
                    sx={{display: results?.audio == null ? "none": "block"}}
                    className="max-w-5xl w-full min-w-full justify-items-center pt-5"
                >
                    <Typography variant='h3' className='pt-2'>
                        { "Images and audio content" }
                    </Typography>

                    <Divider className="pt-2" sx={{width: "95%", height:"2px", color: "#2c243c"}}/>

                    <Card className="w-full pb-5 pl-1 justify-items-center">
                        <ImageDisplay images={results?.image} className="pb-5"/>
                        <Divider className="pt-2" sx={{width: "95%", height:"2px", color: "#2c243c"}}/>
                        <AudioDisplay audioFiles={results?.audio}/>
                    </Card>
                </Card>

                {/* Youtube Videos */}
                <Box
                    sx={{display: results?.audio == null ? "none": "block"}}
                    className="grid-flow-col max-w-5xl w-full min-w-full justify-items-center pt-5 mt-7"
                >
                    <Typography className='w-fit justify-self-start' variant="h4">
                        {"You may find intrest in"}
                        <br/>
                        <i className="pl-7">{"Popular Scientific Content Creators"} </i>
                    </Typography>

                    <Divider className="pt-2" sx={{width: "95%", height:"2px", color: "#2c243c"}}/>

                    <List className="flex" sx={{ listStyle: "decimal", pl: 4 }}>
                        { results?.video?.length > 0 ? (
                            results.video.slice(0,3 ).map((video, index) => (
                                <ResultItemVideo key={index} video={video}/>
                            ))
                        ): ""}
                    </List>
                </Box>
            </Box>
        </div>
    );
}

export default ResultPage;

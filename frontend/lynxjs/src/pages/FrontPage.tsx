import { SearchBar } from "./frontpage/SearchBar.tsx"
import { useState } from '@lynx-js/react'
import { useNavigate,  } from 'react-router';


export function FrontPage() {
    {/* Page Navigation */}
    const nav = useNavigate();
    const [searchQuery, setSearchField] = useState<String>("");

    function Search(e: any) {
        setSearchField(e);
        console.log("Searching with..", e);
        nav(`/search/${encodeURIComponent(e)}`);
    }

    return (
        <view style={{
            position:"relative",
            top:"30%",
            left:"-20%",
            alignContent:"center",
            justifyContent:"center",
            alignSelf:"center",
            justifySelf:"center",
        }}>
            <view className="main" style={{maxWidth: "67vh"}}>
                <text
                    className='Title'
                    style={{fontSize: "3.6rem"}}
                >
                    Welcome to Arxiv API
                </text>
                <SearchBar onsubmit={Search} />
            </view>
        </view>
    );
};
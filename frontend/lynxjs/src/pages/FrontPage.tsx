import { SearchBar } from "./frontpage/SearchBar.tsx"

export function FrontPage() {
    return (
        <view>
            <view className="main">
                <text
                    className='Title'
                    style={{fontSize: "3.6rem"}}
                >
                    Welcome to Arxiv API
                </text>
                <SearchBar/>
            </view>
        </view>
    );
};
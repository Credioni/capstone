// src/App.tsx
import './App.css'

export function App({Page=null} : {Page: any}) {
    function PageHTML(){
        if (Page !== null) {
            return <Page className='Page'/>
        } else {
            return <view className='Page'/>
        }
    }

    return (
        <scroll-view style={{height:"100%", width:"100%"}} scroll-bar-enable enable-scroll>
            <view className='pageContainer'>
                <view className='Background' />
                { PageHTML() }
            </view>
            <view className='footer'>
                <text> Footer </text>
            </view>
        </scroll-view>
    );
}


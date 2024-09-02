export default function Header() {
    return (
        <header className="bg-gray-800 py-4 shadow-md">
            <div className="container mx-auto px-4">
                <div className="flex justify-between items-center">
                    <h1 className="text-white text-2xl font-bold">Stock Prediction App</h1>
                    <nav>
                        <ul className="flex space-x-4">
                            <li><a href="#" className="text-gray-300 hover:text-white transition duration-150 ease-in-out">Home</a></li>
                            <li><a href="#" className="text-gray-300 hover:text-white transition duration-150 ease-in-out">About</a></li>
                            <li><a href="#" className="text-gray-300 hover:text-white transition duration-150 ease-in-out">Contact</a></li>
                        </ul>
                    </nav>
                </div>
            </div>
        </header>
    );
}
export default function Footer() {
    return (
        <footer className="bg-gray-800 py-6 mt-12">
            <div className="container mx-auto px-4">
                <div className="flex flex-col md:flex-row justify-between items-center">
                    <p className="text-white text-sm mb-4 md:mb-0">
                        &copy; {new Date().getFullYear()} Stock Prediction App. All rights reserved.
                    </p>
                    <div className="flex space-x-4">
                        <a href="#" className="text-gray-300 hover:text-white transition duration-150 ease-in-out">
                            Privacy Policy
                        </a>
                        <a href="#" className="text-gray-300 hover:text-white transition duration-150 ease-in-out">
                            Terms of Service
                        </a>
                    </div>
                </div>
            </div>
        </footer>
    );
}
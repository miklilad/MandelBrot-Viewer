#include <iostream>
#include <chrono>
#include "SDL2/SDL.h"
#include "cuda.h"

const int WIDTH = 640; //640
const int HEIGHT = 480; //480

struct HSV
{
    float h;
    float s;
    float v;
};

struct RGB
{
    int r;
    int g;
    int b;
};

RGB Hsv2rgb(const HSV & color)
{
    float s = color.s / 100;
    float v = color.v / 100;
    float c = v * s;
    float h = color.h / 60;
    float x = c * (1 - abs(fmod(h, 2) - 1));
    float m = v - c;
    int hi = (int)floor(h) % 6;
    HSV converted;
    switch(hi)
    {
        case 0: converted = {c, x, 0}; break;
        case 1: converted = {x, c, 0}; break;
        case 2: converted = {0, c, x}; break;
        case 3: converted = {0, x, c}; break;
        case 4: converted = {x, 0, c}; break;
        case 5: converted = {c, 0, x}; break;
        default: return {0, 0, 0};
    }
    int r = (converted.h + m) * 255;
    int g = (converted.s + m) * 255;
    int b = (converted.v + m) * 255;
    return {r, g, b};
}

void Draw(SDL_Renderer * ren, int * data) 
{
    SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
    SDL_RenderClear(ren);
    for (int x = 0; x < WIDTH; x++) 
    {
        for (int y = 0; y < HEIGHT; y++) 
        {
            int i = y * WIDTH + x;
            RGB c = Hsv2rgb({data[i+WIDTH*HEIGHT]+data[i+WIDTH*HEIGHT*2], 100, data[i] % 100});
            SDL_SetRenderDrawColor(ren,c.r, c.g, c.b,255);
            SDL_RenderDrawPoint(ren,x,y);
        }
    }
    SDL_RenderPresent(ren);
}


__global__
void Calculate( int * data, double xC, double yC, double zoom, int iterations, int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = y * width + x;

    if(x >= width || y >= height)
        return;

    data[k+width*height] = blockIdx.x;
    data[k+width*height*2] = blockIdx.y;

    double cR = xC + (x - width / 2) / zoom;
    double cI = yC + (y - height / 2) / zoom;
    double real = 0;
    double imaginary = 0;
    data[k] = 0;
    for (int i = 0; i < iterations; i++) 
    {
        double R = real;
        real = R * R - imaginary * imaginary + cR;
        imaginary = 2 * R * imaginary + cI;
        if (abs(real) > 2 || abs(imaginary) > 2) 
        {
            data[k] = i;
            break;
        }
    }
}


int main() 
{
    int imageSize = HEIGHT * WIDTH * sizeof(int);
    int * data;
    int * d_data;
    data = (int*)malloc(imageSize * 3);
    cudaMalloc(&d_data, imageSize * 3);

    for (int x = 0; x < WIDTH; x++) 
        for (int y = 0; y < HEIGHT; y++) 
            data[y*WIDTH+x] = 255;

    if (SDL_Init(SDL_INIT_VIDEO) != 0) 
    {
        SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
        return 1;
    }

    SDL_Window *win = SDL_CreateWindow("MandelBrot Viewer", 100, 100, WIDTH, HEIGHT, 0);
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);

    int iterations = 1000;
    double x = 0;
    double y = 0;
    double zoom = 150;  //150

    dim3 blockSize(WIDTH/64, HEIGHT/64);

    int bx = (WIDTH + blockSize.x - 1)/blockSize.x;
    int by = (HEIGHT + blockSize.y - 1)/blockSize.y;

    dim3 gridSize(bx, by);

    Calculate<<<gridSize,blockSize>>>(d_data, x, y, zoom, iterations, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    cudaMemcpy(data, d_data, imageSize * 3, cudaMemcpyDeviceToHost);
    Draw(ren, data);



    while (true) 
    {
        SDL_Event event;
        SDL_PollEvent(&event);
        if (event.type == SDL_QUIT)
            break;
        else if (event.type == SDL_KEYDOWN) 
        {
            std::chrono::system_clock::time_point timeStart = std::chrono::system_clock::now();
            switch(event.key.keysym.sym)
            {
                case SDLK_UP:                y -= 1/zoom*5; break;
                case SDLK_DOWN:              y += 1/zoom*5; break;
                case SDLK_RIGHT:             x += 1/zoom*5; break;
                case SDLK_LEFT:              x -= 1/zoom*5; break;
                case SDLK_PAGEUP:         zoom *= 1.1;      break;
                case SDLK_PAGEDOWN:       zoom /= 1.1;      break;
                case SDLK_KP_PLUS:  iterations *= 5;        std::cout << iterations << std::endl; break;
                case SDLK_KP_MINUS: iterations /= 5;        std::cout << iterations << std::endl; break;
                default:                                    break;
            }
            Calculate<<<gridSize,blockSize>>>(d_data, x, y, zoom, iterations, WIDTH, HEIGHT);
            cudaMemcpy(data, d_data, imageSize * 3, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            Draw(ren, data);
            std::chrono::system_clock::time_point timeEnd = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = timeEnd - timeStart;
            double fps = 1 / duration.count();
            std::cout << "FPS: " << fps << std::endl;
            //std::cout << zoom << std::endl; 
        }
    } 

    free(data);
    cudaFree(d_data);

    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
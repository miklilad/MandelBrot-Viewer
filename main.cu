#include <iostream>
#include "SDL2/SDL.h"
#include "cuda.h"


const int WIDTH = 480;
const int HEIGHT = 480;

void Draw(SDL_Renderer * ren, int * data) 
{
    SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
    SDL_RenderClear(ren);
    for (int x = 0; x < WIDTH; x++) 
    {
        for (int y = 0; y < HEIGHT; y++) 
        {
            int i = y * WIDTH + x;
            SDL_SetRenderDrawColor(ren,data[i],data[i],data[i],255);
            SDL_RenderDrawPoint(ren,x,y);
        }
    }
    SDL_RenderPresent(ren);
}


__global__
void Calculate( int * data, double xC, double yC, double zoom, int iterations, int width, int height, int * check)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = y * width + x;

    *check += 1; 

    if(x >= width || y >= height)
        return;

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
    int * data;
    int * d_data;
    data = (int*)malloc(WIDTH * HEIGHT * sizeof(int));
    cudaMalloc(&d_data, WIDTH * HEIGHT * sizeof(int));

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

    Draw(ren,data);
    SDL_Delay(500);

    int iterations = 1000;
    double x = 0;
    double y = 0;
    double zoom = 100;  //150

    dim3 blockSize(WIDTH/16, HEIGHT/16);

    int bx = (WIDTH + blockSize.x - 1)/blockSize.x;
    int by = (HEIGHT + blockSize.y - 1)/blockSize.y;

    dim3 gridSize(bx, by);

    int * check;
    int * d_check;
    check = (int*)malloc(sizeof(int));
    *check = 0;
    cudaMalloc(&d_check, sizeof(int));

        std::cout << *check << std::endl; 

    Calculate<<<gridSize,blockSize>>>(d_data, x, y, zoom, iterations, WIDTH, HEIGHT, d_check);
    cudaDeviceSynchronize();
    cudaMemcpy(data, d_data, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
    cudaMemcpy(check, d_check, sizeof(int), cudaMemcpyDeviceToHost);
    Draw(ren, data);

    std::cout << *check << std::endl; 


    while (true) {
        SDL_Event event;
        SDL_PollEvent(&event);
        if (event.type == SDL_QUIT)
            break;
        else if (event.type == SDL_KEYDOWN) 
        {
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
            Calculate<<<gridSize,blockSize>>>(d_data, x, y, zoom, iterations, WIDTH, HEIGHT, d_check);
            cudaMemcpy(data, d_data, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
            cudaMemcpy(check, d_check, sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            Draw(ren, data);
            std::cout << *check << std::endl; 
        }
    } 

    free(check);
    cudaFree(d_check);

    free(data);
    cudaFree(d_data);

    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
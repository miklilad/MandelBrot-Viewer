#include <iostream>
#include "SDL2/SDL.h"

const int WIDTH = 400;
const int HEIGHT = 400;

void Draw(SDL_Renderer *ren, double xC, double yC, double zoom, int iterations) {
    SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
    SDL_RenderClear(ren);
    for (int x = 0; x < WIDTH; x++) 
    {
        for (int y = 0; y < HEIGHT; y++) 
        {
            double cR = xC + (x - WIDTH / 2) / zoom;
            double cI = yC + (y - HEIGHT / 2) / zoom;
            double real = 0;
            double imaginary = 0;
            for (int i = 0; i < iterations; i++) 
            {
                double R = real;
                real = R * R - imaginary * imaginary + cR;
                imaginary = 2 * R * imaginary + cI;

                if (abs(real) > 2 || abs(imaginary) > 2) 
                {
                    SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
                    break;
                }
            }
            SDL_RenderDrawPoint(ren, x, y);
            SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
        }
    }
    SDL_RenderPresent(ren);
}

int main() 
{
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
    double zoom = 150;
    Draw(ren, x, y, zoom, iterations);

    while (true) {
        SDL_Event event;
        SDL_PollEvent(&event);
        double pX = x;
        double pY = y;
        double pZoom = zoom;
        bool redrawn = true;
        if (event.type == SDL_QUIT)
            break;
        else if (event.type == SDL_KEYDOWN) 
        {
            switch(event.key.keysym.sym)
            {
                case SDLK_UP:          y += 1/zoom; break;
                case SDLK_DOWN:        y -= 1/zoom; break;
                case SDLK_RIGHT:       x += 1/zoom; break;
                case SDLK_LEFT:        x -= 1/zoom; break;
                case SDLK_KP_PLUS:  zoom *= 1.1;    break;
                case SDLK_KP_MINUS: zoom /= 1.1;    break;
                default:                            break;
            }
            std::cout << "Changed" << std::endl;
            Draw(ren, x, y, zoom, iterations);
        }
    }

    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
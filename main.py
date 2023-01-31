from cart_pole import CartPole

if __name__ == "__main__":
    cart_pole = CartPole(num_episodes=500, max_steps=1000, show_graph=True)
    cart_pole.run()
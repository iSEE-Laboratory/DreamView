# astronaut
python launch.py --config configs/dreamview.yaml --train --gpu 0 \
  system.prompt_processor.prompt_global="An astronaut, riding a horse, wearing a white space suit, carrying a red backpack on the back" \
  system.prompt_processor.prompt_front="A horse, an astronaut riding a horse" \
  system.prompt_processor.prompt_right="A bag, red backpack, a horse, an astronaut riding the horse" \
  system.prompt_processor.prompt_back="A bag, red backpack on the astronaut's back, riding a horse" \
  system.prompt_processor.prompt_left="A bag, red backpack, a horse, an astronaut riding the horse" \
  system.prompt_processor.front_threshold=80 system.prompt_processor.back_threshold=80 use_timestamp=False

# bear
python launch.py --config configs/dreamview.yaml --train --gpu 0 \
  system.prompt_processor.prompt_global="A teddy bear toy, holding a basketball and carrying a green backpack on the back, cute, plush" \
  system.prompt_processor.prompt_front="A teddy bear toy, holding a basketball, cute, plush" \
  system.prompt_processor.prompt_right="A teddy bear toy, a green backpack, a basketball" \
  system.prompt_processor.prompt_back="A green backpack on the teddy bear's back, cute, plush" \
  system.prompt_processor.prompt_left="A teddy bear toy, a green backpack, a basketball" \
  system.prompt_processor.front_threshold=70 system.prompt_processor.back_threshold=70 use_timestamp=False

# bulldog
python launch.py --config configs/dreamview.yaml --train --gpu 0 \
  system.prompt_processor.prompt_global="A cute bulldog wearing school uniforms, carrying a rocket on its back, 8K, HD, best quality" \
  system.prompt_processor.prompt_front="A cute bulldog wearing school uniforms, 8K, HD, best quality" \
  system.prompt_processor.prompt_right="A cute bulldog carrying a rocket on its back and wearing school uniforms, 8K, HD, best quality" \
  system.prompt_processor.prompt_back="A cute bulldog carrying a rocket on its back, 8K, HD, best quality" \
  system.prompt_processor.prompt_left="A cute bulldog carrying a rocket on its back and wearing school uniforms, 8K, HD, best quality" \
  system.prompt_processor.front_threshold=70 system.prompt_processor.back_threshold=60 use_timestamp=False

# castle
python launch.py --config configs/dreamview.yaml --train --gpu 0 \
  system.prompt_processor.prompt_global="A castle with a green tree on one side and a red car on the other side, 8K, HD, 3D render, best quality" \
  system.prompt_processor.prompt_front="A green tree locating beside the castle" \
  system.prompt_processor.prompt_right="A castle with a green tree and a red car" \
  system.prompt_processor.prompt_back="A red car locating beside the castle" \
  system.prompt_processor.prompt_left="A castle with a red car and a green tree" \
  system.prompt_processor.front_threshold=80 system.prompt_processor.back_threshold=80 use_timestamp=False

# hulk
python launch.py --config configs/dreamview.yaml --train --gpu 0 \
  system.prompt_processor.prompt_global="The Hulk standing, an AK-47 on his back, ultra realistic, 4k, HD" \
  system.prompt_processor.prompt_front="The Hulk standing, ultra realistic, 4k, HD" \
  system.prompt_processor.prompt_right="A weapon, AK-47, on the back of Hulk" \
  system.prompt_processor.prompt_back="A weapon, gun, AK-47 is on the back of Hulk, ultra realistic, 4k, HD" \
  system.prompt_processor.prompt_left="A weapon, AK-47, on the back of Hulk" \
  system.prompt_processor.front_threshold=85 system.prompt_processor.back_threshold=90 use_timestamp=False

# MAC Book
python launch.py --config configs/dreamview.yaml --train --gpu 0 \
  system.prompt_processor.prompt_global="An open MAC book showing the logo of Superman on the screen" \
  system.prompt_processor.prompt_front="An open MAC book showing the logo of Superman on the screen" \
  system.prompt_processor.prompt_right="Side view of an open MAC book" \
  system.prompt_processor.prompt_back="Back and side view of an open MAC book with an apple logo" \
  system.prompt_processor.prompt_left="Side view of an open MAC book" \
  system.prompt_processor.front_threshold=90 system.prompt_processor.back_threshold=90 use_timestamp=False

# penguin
python launch.py --config configs/dreamview.yaml --train --gpu 0 \
  system.prompt_processor.prompt_global="A penguin wearing a scarf, carrying a crossbody bag on the back, 8K, HD, 3d render, best quality" \
  system.prompt_processor.prompt_front="A penguin wearing a scarf, 8K, HD, 3d render, best quality" \
  system.prompt_processor.prompt_right="A penguin wearing a scarf, carrying a crossbody bag on the back, 8K, HD, 3d render, best quality" \
  system.prompt_processor.prompt_back="A penguin carrying a crossbody bag on the back, 8K, HD, 3d render, best quality" \
  system.prompt_processor.prompt_left="A penguin wearing a scarf, carrying a crossbody bag on the back, 8K, HD, 3d render, best quality" \
  system.prompt_processor.front_threshold=60 system.prompt_processor.back_threshold=80 use_timestamp=False

# Pikachu
python launch.py --config configs/dreamview.yaml --train --gpu 0 \
  system.prompt_processor.prompt_global="Pikachu, wearing a top hat, red flame on its tail" \
  system.prompt_processor.prompt_front="Pikachu, wearing a top hat" \
  system.prompt_processor.prompt_right="Red flame, fire, on Pikachu's tail, a top hat" \
  system.prompt_processor.prompt_back="fire, red flame on tail, Pikachu wearing a top hat" \
  system.prompt_processor.prompt_left="Red flame, fire, on Pikachu's tail, a top hat" \
  system.prompt_processor.front_threshold=60 system.prompt_processor.back_threshold=80 use_timestamp=False

# T-shirt
python launch.py --config configs/dreamview.yaml --train --gpu 0 \
  system.prompt_processor.prompt_global="A T-shirt with a sun on the front and some letters on the back" \
  system.prompt_processor.prompt_front="A T-shirt with a sun on it" \
  system.prompt_processor.prompt_right="A T-shirt" \
  system.prompt_processor.prompt_back="A T-shirt with some letters" \
  system.prompt_processor.prompt_left="A T-shirt" \
  system.prompt_processor.front_threshold=80 system.prompt_processor.back_threshold=80 use_timestamp=False
